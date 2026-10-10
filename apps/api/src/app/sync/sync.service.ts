import { Injectable, Logger } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { z } from 'zod';
import type { Case, Patient, RevisitAssessment, TherapyDetails } from '@antigravity-project-spec-pack/domain';
import type { Care, IntakeAnswers, Progress } from '@antigravity-project-spec-pack/domain/wound-model';
import {
  SYNC_ENTITIES,
  type PullResponse,
  type PushResponse,
  type Rejected,
  type SyncEntity,
  type SyncTreatment,
} from '@antigravity-project-spec-pack/domain/sync';
import type { ClinicContext } from '../auth/clinic.guard';
import { SummaryService } from '../cases/summary.service';
import { searchTextOf } from '../patients/patients.service';
import { AuditService } from '../platform/audit.service';
import { CacheService } from '../platform/cache.service';
import { parse } from '../platform/validation';
import { PrismaService } from '../prisma.service';
import { StorageService } from '../visits/storage.service';
import { VisitsService, type PhotoUpload } from '../visits/visits.service';

const PAGE = 500;
/** Pull overlaps the previous one by this much, so a write committed just after a pull started is never missed. */
const OVERLAP_MS = 5000;
const day = z.string().regex(/^\d{4}-\d{2}-\d{2}/);
const toDay = (v: string) => new Date(`${v.slice(0, 10)}T00:00:00Z`);
const iso = (d: Date | null | undefined) => (d ? d.toISOString() : undefined);
const dayOf = (d: Date | null | undefined) => (d ? d.toISOString().slice(0, 10) : '');

// Records are validated one by one on the server, so one bad record never blocks the rest of a push.
const PatientRecord = z.object({
  id: z.uuid(),
  firstName: z.string().trim().min(1).max(100),
  lastName: z.string().trim().max(100),
  patientId: z.string().trim().max(64),
  sex: z.enum(['M', 'F', 'O']),
  dob: z.union([day, z.literal('')]).optional(),
  location: z.string().max(200).optional(),
  consent: z.object({ care: z.boolean(), aiTraining: z.boolean(), noticeVersion: z.string().max(40), recordedAt: z.string().max(40) }).optional(),
});
const CaseRecord = z.object({
  id: z.uuid(),
  patientId: z.uuid(),
  onsetDate: day,
  woundLocation: z.string().trim().min(1).max(200),
  status: z.enum(['IN_TREATMENT', 'EVALUATION', 'COMPLETED']),
  createdAt: z.string().max(40).optional(),
});
const answers = z.array(z.looseObject({})).max(100).optional();
const TreatmentRecord = z.object({
  id: z.uuid(),
  caseId: z.uuid(),
  sequenceNumber: z.number().int().min(1).max(10000),
  phase: z.enum(['PRE', 'POST', 'COMPLETED']),
  assessment: z
    .object({
      woundType: z.string().max(100).optional().default(''),
      exudateLevel: z.string().max(60).optional().default(''),
      exudateType: z.string().max(60).optional().default(''),
      infectionSigns: z.array(z.string().max(100)).max(30).optional().default([]),
      pain: z.number().min(0).max(10).optional(),
      edgeCondition: z.string().max(100).optional().default(''),
      periwoundCondition: z.string().max(100).optional().default(''),
      comorbidities: z.array(z.string().max(100)).max(30).optional().default([]),
      woundAppearanceTrend: z.enum(['Improving', 'Static', 'Deteriorating']).optional(),
      responses: answers,
    })
    .optional(),
  therapy: z
    .object({
      therapyGiven: z.array(z.string().max(120)).max(40).default([]),
      dressingType: z.string().max(120).default(''),
      nextVisitDate: day.optional(),
      responses: answers,
    })
    .optional(),
  createdAt: z.string().max(40).optional(),
});

const PushBody = z.object({
  deviceId: z.string().min(1).max(100),
  changes: z.partialRecord(
    z.enum(SYNC_ENTITIES),
    z.object({
      upserted: z.array(z.object({ record: z.looseObject({ id: z.string() }), changed: z.array(z.string().max(60)).max(60) })).max(1000).default([]),
      deleted: z.array(z.string()).max(1000).default([]),
    }),
  ),
});

type Cursors = Partial<Record<SyncEntity, { t: string; id: string }>>;
const encode = (c: Cursors) => Buffer.from(JSON.stringify(c)).toString('base64url');
const decode = (raw: unknown): Cursors => {
  if (typeof raw !== 'string' || !raw) return {};
  try {
    return JSON.parse(Buffer.from(raw, 'base64url').toString('utf8')) as Cursors;
  } catch {
    return {};
  }
};

// Where on the body, in the model's words, from the wound's location as the clinician typed it.
const SITES: [RegExp, string][] = [
  [/\bheel/i, 'heel'],
  [/\btoe/i, 'toe'],
  [/\bankle|malleol/i, 'ankle'],
  [/\bknee/i, 'knee'],
  [/\bthigh/i, 'thigh'],
  [/\b(lower leg|calf|shin|leg)/i, 'lower_leg'],
  [/\b(sole|plantar)/i, 'foot_plantar'],
  [/\bfoot|dorsum/i, 'foot_dorsal'],
  [/\b(sacr|buttock|coccyx|gluteal)/i, 'sacrum_buttock'],
  [/\bhip|trochanter/i, 'hip'],
  [/\bback|spine|scapula/i, 'back'],
  [/\babdom/i, 'abdomen'],
  [/\bchest|breast/i, 'chest'],
  [/\bhand|finger|wrist/i, 'hand'],
  [/\barm|elbow|forearm/i, 'arm'],
  [/\bhead|neck|face|scalp|ear/i, 'head_neck'],
];
const CAUSES: [RegExp, string][] = [
  [/pressure/i, 'pressure_lying_or_sitting'],
  [/burn/i, 'burn'],
  [/surg/i, 'surgery'],
  [/trauma|injur|lacerat|cut|fall/i, 'injury_cut_or_fall'],
];

/**
 * The model's required answers for a photo from the app, which asks its own clinical questions instead:
 * body site from the wound's location, diabetes from the comorbidities, cause from the wound type.
 * Unknowns are sent as "not sure"/"unknown" rather than guessed.
 */
export function intakeFromApp(location: string, a: Partial<RevisitAssessment> | null, caseComorbidities: string[]): IntakeAnswers {
  const comorbidities = [...(a?.comorbidities ?? []), ...caseComorbidities];
  const type = a?.woundType || '';
  return {
    body_location: SITES.find(([re]) => re.test(location))?.[1] ?? 'other',
    diabetes: comorbidities.some((c) => /diabet/i.test(c)) || /diabet/i.test(type) ? 'yes' : 'not_sure',
    cause: CAUSES.find(([re]) => re.test(type))?.[1] ?? 'unknown',
    ...(typeof a?.pain === 'number' ? { pain: a.pain } : {}),
  } as IntakeAnswers;
}

/** Offline-first sync for the mobile app: field-level merge on push, incremental pull. */
@Injectable()
export class SyncService {
  private readonly logger = new Logger(SyncService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly summary: SummaryService,
    private readonly cache: CacheService,
    private readonly audit: AuditService,
    private readonly storage: StorageService,
    private readonly visits: VisitsService,
  ) {}

  // ---------------------------------------------------------------- push

  async push(ctx: ClinicContext, raw: unknown): Promise<PushResponse> {
    const body = parse(PushBody, raw, 'Sync request not understood.');
    const rejected: Rejected[] = [];
    const touchedCases = new Set<string>();
    const touchedPatients = new Set<string>();
    const clinical = ctx.role !== 'FRONT_DESK';
    let applied = 0;

    // Parents first, so a new patient, its wound and its first treatment can arrive in one push.
    for (const entity of SYNC_ENTITIES) {
      const part = body.changes[entity];
      if (!part) continue;
      for (const { record, changed } of part.upserted) {
        const id = String(record.id);
        if (entity !== 'patients' && !clinical) {
          rejected.push({ entity, id, reason: 'Your role does not include clinical records.' });
          continue;
        }
        try {
          const reason = await this.upsert(ctx, entity, record, changed, touchedCases, touchedPatients);
          if (reason) rejected.push({ entity, id, reason });
          else applied++;
        } catch (error) {
          this.logger.warn(`Sync ${entity} ${id} failed: ${error instanceof Error ? error.message : String(error)}`);
          rejected.push({ entity, id, reason: 'Could not be saved. It will be retried.' });
        }
      }
      for (const id of part.deleted) {
        if (entity !== 'patients' && !clinical) {
          rejected.push({ entity, id, reason: 'Your role does not include clinical records.' });
          continue;
        }
        await this.remove(ctx, entity, id, touchedCases, touchedPatients);
        applied++;
      }
    }

    for (const caseId of touchedCases) await this.summary.refreshCase(caseId);
    for (const patientId of touchedPatients) await this.summary.refreshPatient(patientId);
    if (applied) {
      await Promise.all([
        this.cache.bump(ctx.clinicId),
        this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'sync.push', details: { deviceId: body.deviceId, records: applied, rejected: rejected.length } }),
      ]);
    }
    return { serverTime: new Date().toISOString(), rejected };
  }

  /** Applies one record. Returns why it was refused, or null. */
  private async upsert(
    ctx: ClinicContext,
    entity: SyncEntity,
    raw: unknown,
    changed: string[],
    touchedCases: Set<string>,
    touchedPatients: Set<string>,
  ): Promise<string | null> {
    const has = (f: string) => changed.includes(f);
    if (entity === 'patients') {
      const r = PatientRecord.safeParse(raw);
      if (!r.success) return `Invalid: ${r.error.issues.map((i) => i.path.join('.')).join(', ')}`;
      const p = r.data;
      const existing = await this.prisma.patient.findUnique({ where: { id: p.id }, select: { clinicId: true, deletedAt: true, firstName: true, lastName: true, patientId: true } });
      if (existing && existing.clinicId !== ctx.clinicId) return 'Not found.';
      if (existing?.deletedAt) return 'This patient was deleted.';
      const consent = p.consent ? { ...p.consent, photos: p.consent.care, location: false } : undefined;
      if (!existing) {
        const code = await this.freeCode(ctx.clinicId, p.patientId);
        await this.prisma.patient.create({
          data: {
            id: p.id,
            clinicId: ctx.clinicId,
            createdById: ctx.userId,
            patientId: code,
            firstName: p.firstName,
            lastName: p.lastName,
            sex: p.sex,
            dateOfBirth: p.dob ? toDay(p.dob) : null,
            location: p.location || null,
            consent: consent as Prisma.InputJsonValue | undefined,
            searchText: searchTextOf({ firstName: p.firstName, lastName: p.lastName, patientId: code }),
          },
        });
        return null;
      }
      const data: Prisma.PatientUpdateInput = { version: { increment: 1 } };
      if (has('firstName')) data.firstName = p.firstName;
      if (has('lastName')) data.lastName = p.lastName;
      if (has('sex')) data.sex = p.sex;
      if (has('dob')) data.dateOfBirth = p.dob ? toDay(p.dob) : null;
      if (has('location')) data.location = p.location || null;
      if (has('consent') && consent) data.consent = consent as Prisma.InputJsonValue;
      let code = existing.patientId;
      if (has('patientId') && p.patientId.trim().toUpperCase() !== existing.patientId) {
        code = await this.freeCode(ctx.clinicId, p.patientId, p.id);
        data.patientId = code;
      }
      data.searchText = searchTextOf({
        firstName: (data.firstName as string | undefined) ?? existing.firstName,
        lastName: (data.lastName as string | undefined) ?? existing.lastName,
        patientId: code,
      });
      await this.prisma.patient.update({ where: { id: p.id }, data });
      return null;
    }

    if (entity === 'cases') {
      const r = CaseRecord.safeParse(raw);
      if (!r.success) return `Invalid: ${r.error.issues.map((i) => i.path.join('.')).join(', ')}`;
      const c = r.data;
      const existing = await this.prisma.case.findUnique({ where: { id: c.id }, select: { clinicId: true, deletedAt: true, patientId: true } });
      if (existing && existing.clinicId !== ctx.clinicId) return 'Not found.';
      if (existing?.deletedAt) return 'This wound was deleted.';
      if (!existing) {
        const patient = await this.prisma.patient.findFirst({ where: { id: c.patientId, clinicId: ctx.clinicId, deletedAt: null }, select: { id: true } });
        if (!patient) return 'Its patient is missing or was deleted.';
        await this.prisma.case.create({
          data: {
            id: c.id,
            clinicId: ctx.clinicId,
            patientId: c.patientId,
            onset: toDay(c.onsetDate),
            location: c.woundLocation,
            woundType: 'Not recorded',
            comorbidities: [],
            closedAt: c.status === 'COMPLETED' ? new Date() : null,
            ...(c.createdAt ? { createdAt: new Date(c.createdAt) } : {}),
          },
        });
      } else {
        const data: Prisma.CaseUpdateInput = { version: { increment: 1 } };
        if (has('woundLocation')) data.location = c.woundLocation;
        if (has('onsetDate')) data.onset = toDay(c.onsetDate);
        if (has('status')) data.closedAt = c.status === 'COMPLETED' ? new Date() : null;
        await this.prisma.case.update({ where: { id: c.id }, data });
      }
      touchedCases.add(c.id);
      touchedPatients.add(c.patientId);
      return null;
    }

    const r = TreatmentRecord.safeParse(raw);
    if (!r.success) return `Invalid: ${r.error.issues.map((i) => i.path.join('.')).join(', ')}`;
    const t = r.data;
    const existing = await this.prisma.treatment.findUnique({ where: { id: t.id }, select: { clinicId: true, deletedAt: true, caseId: true } });
    if (existing && existing.clinicId !== ctx.clinicId) return 'Not found.';
    if (existing?.deletedAt) return 'This treatment was deleted.';
    const woundCase = await this.prisma.case.findFirst({
      where: { id: existing?.caseId ?? t.caseId, clinicId: ctx.clinicId, deletedAt: null },
      select: { id: true, patientId: true, woundType: true, comorbidities: true },
    });
    if (!woundCase) return 'Its wound is missing or was deleted.';

    const therapy = t.therapy;
    const therapyData = {
      therapy: therapy?.therapyGiven ?? [],
      dressing: therapy?.dressingType || null,
      nextVisit: therapy?.nextVisitDate ? toDay(therapy.nextVisitDate) : null,
      careResponses: (therapy?.responses ?? undefined) as Prisma.InputJsonValue | undefined,
    };
    await this.prisma.$transaction(async (tx) => {
      if (!existing) {
        // Two offline devices may both have made "T3": the later one becomes the next number.
        const taken = await tx.treatment.findFirst({ where: { caseId: woundCase.id, sequence: t.sequenceNumber, deletedAt: null }, select: { id: true } });
        const last = taken ? await tx.treatment.findFirst({ where: { caseId: woundCase.id }, orderBy: { sequence: 'desc' }, select: { sequence: true } }) : null;
        await tx.treatment.create({
          data: {
            id: t.id,
            clinicId: ctx.clinicId,
            caseId: woundCase.id,
            sequence: taken ? (last?.sequence ?? 0) + 1 : t.sequenceNumber,
            ...therapyData,
            ...(t.createdAt ? { createdAt: new Date(t.createdAt) } : {}),
          },
        });
      } else if (has('therapy')) {
        await tx.treatment.update({ where: { id: t.id }, data: { ...therapyData, version: { increment: 1 } } });
      } else {
        await tx.treatment.update({ where: { id: t.id }, data: { version: { increment: 1 } } });
      }

      if (t.assessment && (!existing || has('assessment'))) {
        const a = t.assessment;
        const phase =
          (await tx.phase.findFirst({ where: { treatmentId: t.id, phaseType: 'PRE' }, select: { id: true } })) ??
          (await tx.phase.create({ data: { clinicId: ctx.clinicId, treatmentId: t.id, phaseType: 'PRE' }, select: { id: true } }));
        const fields = {
          exudateLevel: a.exudateLevel || 'Not recorded',
          exudateType: a.exudateType || null,
          infectionSigns: a.infectionSigns,
          painLevel: a.pain ?? 0,
          edgeCondition: a.edgeCondition || null,
          periwoundCondition: a.periwoundCondition || null,
          trend: a.woundAppearanceTrend ?? null,
          responses: (a.responses ?? undefined) as Prisma.InputJsonValue | undefined,
        };
        await tx.clinicalAssessment.upsert({ where: { phaseId: phase.id }, create: { phaseId: phase.id, ...fields }, update: fields });
        // The wound's type and comorbidities come from the clinician's latest assessment.
        const caseData: Prisma.CaseUpdateInput = {};
        if (a.woundType) caseData.woundType = a.woundType;
        const union = [...new Set([...woundCase.comorbidities, ...a.comorbidities])];
        if (union.length !== woundCase.comorbidities.length) caseData.comorbidities = union;
        if (Object.keys(caseData).length) await tx.case.update({ where: { id: woundCase.id }, data: { ...caseData, version: { increment: 1 } } });
      }
    });
    touchedCases.add(woundCase.id);
    touchedPatients.add(woundCase.patientId);
    return null;
  }

  /** A patient code that is free in the clinic: the one asked for, with -2, -3… if taken; WM-0001… if blank. */
  private async freeCode(clinicId: string, wanted: string, selfId?: string): Promise<string> {
    const base = wanted.trim().toUpperCase();
    if (!base) {
      const clinic = await this.prisma.clinic.update({ where: { id: clinicId }, data: { patientSeq: { increment: 1 } }, select: { patientSeq: true } });
      return `WM-${String(clinic.patientSeq).padStart(4, '0')}`;
    }
    const taken = await this.prisma.patient.findMany({
      where: { clinicId, patientId: { startsWith: base }, ...(selfId ? { id: { not: selfId } } : {}) },
      select: { patientId: true },
    });
    const used = new Set(taken.map((p) => p.patientId));
    if (!used.has(base)) return base;
    for (let n = 2; ; n++) if (!used.has(`${base}-${n}`)) return `${base}-${n}`;
  }

  private async remove(ctx: ClinicContext, entity: SyncEntity, id: string, touchedCases: Set<string>, touchedPatients: Set<string>) {
    const now = new Date();
    const data = { deletedAt: now, version: { increment: 1 } };
    if (entity === 'patients') {
      const p = await this.prisma.patient.findFirst({ where: { id, clinicId: ctx.clinicId, deletedAt: null }, select: { id: true } });
      if (!p) return;
      const photos = await this.prisma.image.findMany({ where: { phase: { treatment: { case: { patientId: id } } } }, select: { imageUrl: true, thumbPath: true } });
      await this.prisma.$transaction([
        this.prisma.patient.update({ where: { id }, data }),
        this.prisma.case.updateMany({ where: { patientId: id, deletedAt: null }, data: { deletedAt: now } }),
      ]);
      await this.storage.remove(photos.flatMap((i) => [i.imageUrl, i.thumbPath ?? '']));
      await this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'patient.delete', entity: 'Patient', entityId: id, details: { from: 'app' } });
    } else if (entity === 'cases') {
      const c = await this.prisma.case.findFirst({ where: { id, clinicId: ctx.clinicId, deletedAt: null }, select: { patientId: true } });
      if (!c) return;
      const photos = await this.prisma.image.findMany({ where: { phase: { treatment: { caseId: id } } }, select: { imageUrl: true, thumbPath: true } });
      await this.prisma.case.update({ where: { id }, data });
      await this.storage.remove(photos.flatMap((i) => [i.imageUrl, i.thumbPath ?? '']));
      touchedPatients.add(c.patientId);
    } else {
      const t = await this.prisma.treatment.findFirst({ where: { id, clinicId: ctx.clinicId, deletedAt: null }, select: { caseId: true, case: { select: { patientId: true } } } });
      if (!t) return;
      const photos = await this.prisma.image.findMany({ where: { phase: { treatmentId: id } }, select: { imageUrl: true, thumbPath: true } });
      await this.prisma.$transaction([
        this.prisma.aIResult.deleteMany({ where: { phase: { treatmentId: id } } }),
        this.prisma.image.deleteMany({ where: { phase: { treatmentId: id } } }),
        this.prisma.treatment.update({ where: { id }, data }),
      ]);
      await this.storage.remove(photos.flatMap((i) => [i.imageUrl, i.thumbPath ?? '']));
      touchedCases.add(t.caseId);
      touchedPatients.add(t.case.patientId);
    }
  }

  // ---------------------------------------------------------------- pull

  async pull(ctx: ClinicContext, sinceRaw: unknown, cursorRaw: unknown): Promise<PullResponse> {
    const serverTime = new Date().toISOString();
    const parsed = typeof sinceRaw === 'string' && sinceRaw ? new Date(sinceRaw) : null;
    const since = parsed && !Number.isNaN(parsed.getTime()) ? new Date(parsed.getTime() - OVERLAP_MS) : new Date(0);
    const cursors = decode(cursorRaw);
    const next: Cursors = {};
    const clinical = ctx.role !== 'FRONT_DESK';

    // Rows changed after `since`, oldest first, after this entity's cursor; a full page means more are waiting.
    const window = (entity: SyncEntity) => {
      const c = cursors[entity];
      return {
        clinicId: ctx.clinicId,
        updatedAt: { gt: since },
        ...(c ? { OR: [{ updatedAt: { gt: new Date(c.t) } }, { updatedAt: new Date(c.t), id: { gt: c.id } }] } : {}),
      };
    };
    const order = [{ updatedAt: 'asc' as const }, { id: 'asc' as const }];
    const mark = <T extends { id: string; updatedAt: Date }>(entity: SyncEntity, rows: T[]) => {
      if (rows.length === PAGE) {
        const last = rows[rows.length - 1];
        next[entity] = { t: last.updatedAt.toISOString(), id: last.id };
      }
      return rows;
    };

    const [patients, cases, treatments] = await Promise.all([
      this.prisma.patient
        .findMany({ where: window('patients'), orderBy: order, take: PAGE })
        .then((rows) => mark('patients', rows)),
      clinical ? this.prisma.case.findMany({ where: window('cases'), orderBy: order, take: PAGE }).then((rows) => mark('cases', rows)) : Promise.resolve([]),
      clinical
        ? this.prisma.treatment
            .findMany({
              where: window('treatments'),
              orderBy: order,
              take: PAGE,
              include: {
                phases: {
                  select: {
                    phaseType: true,
                    assessment: true,
                    image: { select: { imageUrl: true, thumbPath: true } },
                    aiResult: { select: { id: true, status: true, area: true, urgent: true, reviewStatus: true, findings: true, progress: true, care: true } },
                  },
                },
                case: { select: { woundType: true, comorbidities: true } },
              },
            })
            .then((rows) => mark('treatments', rows))
        : Promise.resolve([]),
    ]);

    // Paging continues: entities that are already complete are marked done so they aren't sent again.
    if (Object.keys(next).length) {
      for (const e of SYNC_ENTITIES) if (!next[e]) next[e] = { t: '9999-12-31T00:00:00.000Z', id: '' };
    }

    const paths = treatments.flatMap((t) => t.phases.map((p) => p.image?.thumbPath ?? p.image?.imageUrl ?? '')).filter(Boolean);
    const urls = await this.storage.signedUrls(paths);
    const live = <T extends { deletedAt: Date | null }>(rows: T[]) => rows.filter((r) => !r.deletedAt);
    const gone = <T extends { id: string; deletedAt: Date | null }>(rows: T[]) => rows.filter((r) => r.deletedAt).map((r) => r.id);

    return {
      serverTime,
      cursor: Object.keys(next).length ? encode({ ...cursors, ...next }) : null,
      changes: {
        patients: {
          upserted: live(patients).map(
            (p): Omit<Patient, 'syncState'> => ({
              id: p.id,
              firstName: p.firstName,
              lastName: p.lastName,
              patientId: p.patientId,
              sex: p.sex,
              dob: dayOf(p.dateOfBirth),
              location: p.location ?? '',
              ...(p.consent ? { consent: p.consent as unknown as Patient['consent'] } : {}),
            }),
          ),
          deleted: gone(patients),
        },
        cases: {
          upserted: live(cases).map(
            (c): Omit<Case, 'syncState'> => ({
              id: c.id,
              patientId: c.patientId,
              onsetDate: dayOf(c.onset),
              woundLocation: c.location,
              status: c.closedAt ? 'COMPLETED' : 'IN_TREATMENT',
              createdAt: c.createdAt.toISOString(),
            }),
          ),
          deleted: gone(cases),
        },
        treatments: {
          upserted: live(treatments).map((t) => this.toSyncTreatment(t, urls)),
          deleted: gone(treatments),
        },
      },
    };
  }

  private toSyncTreatment(
    t: Prisma.TreatmentGetPayload<{
      include: {
        phases: {
          select: {
            phaseType: true;
            assessment: true;
            image: { select: { imageUrl: true; thumbPath: true } };
            aiResult: { select: { id: true; status: true; area: true; urgent: true; reviewStatus: true; findings: true; progress: true; care: true } };
          };
        };
        case: { select: { woundType: true; comorbidities: true } };
      };
    }>,
    urls: Map<string, string | null>,
  ): SyncTreatment {
    const pre = t.phases.find((p) => p.phaseType === 'PRE');
    const post = t.phases.find((p) => p.phaseType === 'POST');
    const a = pre?.assessment;
    const url = (img: { imageUrl: string; thumbPath: string | null } | null | undefined) => (img ? urls.get(img.thumbPath ?? img.imageUrl) ?? null : null);
    const ai = pre?.aiResult;
    const findings = ai?.findings as { wound_type?: { label?: string } } | null;
    const healing = (ai?.progress as Progress | null)?.healing;
    // Suggestions reach the app only once a clinician has approved or edited the draft.
    const approved = ai?.reviewStatus === 'approved' || ai?.reviewStatus === 'edited';
    const care = approved ? (ai?.care as Care | null) : null;
    const therapy: TherapyDetails | undefined =
      t.therapy.length || t.dressing || t.nextVisit || t.careResponses
        ? {
            therapyGiven: t.therapy,
            dressingType: t.dressing ?? '',
            ...(t.nextVisit ? { nextVisitDate: dayOf(t.nextVisit) } : {}),
            ...(t.careResponses ? { responses: t.careResponses as unknown as TherapyDetails['responses'] } : {}),
          }
        : undefined;
    const assessment: RevisitAssessment | undefined = a
      ? {
          woundType: t.case.woundType === 'Not recorded' ? '' : t.case.woundType,
          exudateLevel: a.exudateLevel === 'Not recorded' ? '' : a.exudateLevel,
          exudateType: a.exudateType ?? '',
          infectionSigns: a.infectionSigns,
          pain: a.painLevel,
          edgeCondition: a.edgeCondition ?? '',
          periwoundCondition: a.periwoundCondition ?? '',
          comorbidities: t.case.comorbidities,
          ...(a.trend ? { woundAppearanceTrend: a.trend as RevisitAssessment['woundAppearanceTrend'] } : {}),
          ...(a.responses ? { responses: a.responses as unknown as RevisitAssessment['responses'] } : {}),
        }
      : undefined;
    return {
      id: t.id,
      caseId: t.caseId,
      sequenceNumber: t.sequence,
      // The app's flow: assessment done → waiting for the post photo; care recorded → complete.
      phase: therapy ? 'COMPLETED' : assessment ? 'POST' : 'PRE',
      ...(assessment ? { assessment } : {}),
      ...(therapy ? { therapy } : {}),
      createdAt: t.createdAt.toISOString(),
      remote: {
        preStored: !!pre?.image,
        postStored: !!post?.image,
        preUrl: url(pre?.image),
        postUrl: url(post?.image),
        ai: ai
          ? {
              visitId: ai.id,
              status: ai.status as NonNullable<NonNullable<SyncTreatment['remote']>['ai']>['status'],
              areaCm2: ai.area,
              woundType: findings?.wound_type?.label ?? null,
              urgent: ai.urgent,
              review: (ai.reviewStatus === 'pending' ? null : ai.reviewStatus) as 'approved' | 'edited' | 'rejected' | null,
              postStatus: (post?.aiResult?.status ?? null) as NonNullable<NonNullable<SyncTreatment['remote']>['ai']>['postStatus'],
              trajectory: healing?.trajectory ?? null,
              areaReductionSinceFirstPct: healing?.since_first?.percent_area_reduction ?? null,
              suggestions: care ? care.suggestions.map((s) => ({ action: s.action, text: s.text })) : null,
            }
          : null,
      },
    };
  }

  // ---------------------------------------------------------------- photos

  /** A photo taken in the app, sent after its treatment synced. Both photos are analysed by the model. */
  async photo(ctx: ClinicContext, treatmentId: string, phaseRaw: string, photo: PhotoUpload | undefined) {
    const phase = phaseRaw.toUpperCase() === 'POST' ? 'POST' : 'PRE';
    let intake: IntakeAnswers = {} as IntakeAnswers;
    {
      // The same answers for both photos of a visit: the POST photo is the same wound minutes later.
      const t = await this.prisma.treatment.findFirst({
        where: { id: treatmentId, clinicId: ctx.clinicId },
        select: { case: { select: { location: true, comorbidities: true, woundType: true } }, phases: { where: { phaseType: 'PRE' }, select: { assessment: true } } },
      });
      const a = t?.phases[0]?.assessment;
      intake = intakeFromApp(
        t?.case.location ?? '',
        a ? { woundType: t?.case.woundType ?? '', comorbidities: [], pain: a.painLevel } : { woundType: t?.case.woundType ?? '' },
        t?.case.comorbidities ?? [],
      );
    }
    const result = await this.visits.attachSyncedPhoto(ctx, treatmentId, phase, photo, intake);
    // Other devices learn the photo is stored on their next pull.
    await this.prisma.treatment.update({ where: { id: treatmentId }, data: { version: { increment: 1 } } });
    return result;
  }
}
