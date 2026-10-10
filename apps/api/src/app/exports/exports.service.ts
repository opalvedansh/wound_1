import { BadRequestException, Injectable } from '@nestjs/common';
import type { ClinicContext } from '../auth/clinic.guard';
import { AuditService } from '../platform/audit.service';
import { PrismaService } from '../prisma.service';
import { PRE_VISIT } from '../views';

const BATCH = 1000;

export const csvCell = (value: unknown): string => {
  if (value === null || value === undefined) return '';
  const s = value instanceof Date ? value.toISOString() : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
export const csvRow = (values: unknown[]) => `${values.map(csvCell).join(',')}\n`;

export const ageBand = (age: number | null) => (age === null ? '' : `${Math.floor(age / 10) * 10}-${Math.floor(age / 10) * 10 + 9}`);
const month = (d: Date | null) => (d ? d.toISOString().slice(0, 7) : '');
const dayOf = (d: Date | null) => (d ? d.toISOString().slice(0, 10) : '');

const ageOf = (dob: Date | null, ageYears: number | null, now = new Date()) => {
  if (!dob) return ageYears;
  let age = now.getUTCFullYear() - dob.getUTCFullYear();
  const m = now.getUTCMonth() - dob.getUTCMonth();
  if (m < 0 || (m === 0 && now.getUTCDate() < dob.getUTCDate())) age--;
  return age;
};

export type Dataset = 'patients' | 'visits' | 'validation';

const joined = (values: string[] | null | undefined) => (values ?? []).join(';');
const json = (value: unknown) => (value === null || value === undefined ? '' : JSON.stringify(value));

type ModelFindings = {
  wound_type?: { label?: string; prob?: number };
  severity?: Record<string, { label?: string; prob?: number }>;
  tissue_pct?: Record<string, number>;
  tissue_untrusted?: string[];
  periwound_erythema_frac?: number;
  periwound_maceration_frac?: number;
  periwound_callus_frac?: number;
  flags_version?: string;
};

/**
 * CSV exports, streamed in batches with keyset pagination, so a clinic of any size exports in constant memory.
 * De-identified by default: no names, contact details, exact birth dates or exact dates (months instead).
 * Each export is written to the audit log before any data leaves ("no log entry, no file").
 */
@Injectable()
export class ExportsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly audit: AuditService,
  ) {}

  async *stream(ctx: ClinicContext, dataset: string, deidentify: boolean): AsyncGenerator<string> {
    if (dataset !== 'patients' && dataset !== 'visits' && dataset !== 'validation') {
      throw new BadRequestException('Unknown dataset. Use "patients", "visits" or "validation".');
    }
    // The validation set is for scoring the model outside the clinic: always de-identified.
    const deid = dataset === 'validation' || deidentify;
    await this.audit.logNow({ clinicId: ctx.clinicId, userId: ctx.userId, action: `export.${dataset}`, details: { deidentified: deid } });
    if (dataset === 'patients') yield* this.patients(ctx, deid);
    else if (dataset === 'visits') yield* this.visits(ctx, deid);
    else yield* this.validation(ctx);
  }

  private async *patients(ctx: ClinicContext, deid: boolean): AsyncGenerator<string> {
    yield csvRow(
      deid
        ? ['patient_code', 'sex', 'age_band', 'status', 'open_wounds', 'registered_month']
        : ['patient_code', 'first_name', 'last_name', 'sex', 'age', 'date_of_birth', 'mobile', 'location', 'status', 'open_wounds', 'registered'],
    );
    let after: string | undefined;
    for (;;) {
      const rows = await this.prisma.patient.findMany({
        where: { clinicId: ctx.clinicId, deletedAt: null, ...(after ? { id: { gt: after } } : {}) },
        orderBy: { id: 'asc' },
        take: BATCH,
        include: { _count: { select: { cases: { where: { deletedAt: null, closedAt: null } } } } },
      });
      if (rows.length === 0) return;
      let chunk = '';
      for (const p of rows) {
        const age = ageOf(p.dateOfBirth, p.ageYears);
        chunk += deid
          ? csvRow([p.patientId, p.sex, ageBand(age), p.status, p._count.cases, month(p.createdAt)])
          : csvRow([p.patientId, p.firstName, p.lastName, p.sex, age, dayOf(p.dateOfBirth), p.mobile, p.location, p.status, p._count.cases, dayOf(p.createdAt)]);
      }
      yield chunk;
      after = rows[rows.length - 1].id;
    }
  }

  private async *visits(ctx: ClinicContext, deid: boolean): AsyncGenerator<string> {
    yield csvRow([
      'patient_code',
      ...(deid ? [] : ['patient_name']),
      'case_id',
      'wound_location',
      'wound_type',
      'visit',
      deid ? 'visit_month' : 'visit_date',
      'area_cm2',
      'length_cm',
      'width_cm',
      'model_wound_type',
      'model_confidence',
      'urgent',
      'flags',
      'review',
      'model_versions',
    ]);
    let after: string | undefined;
    for (;;) {
      const rows = await this.prisma.aIResult.findMany({
        where: { clinicId: ctx.clinicId, status: 'ok', ...PRE_VISIT, ...(after ? { id: { gt: after } } : {}) },
        orderBy: { id: 'asc' },
        take: BATCH,
        select: {
          id: true,
          createdAt: true,
          area: true,
          length: true,
          height: true,
          confidenceScore: true,
          urgent: true,
          flagCount: true,
          reviewStatus: true,
          modelVersions: true,
          findings: true,
          phase: {
            select: {
              treatment: {
                select: {
                  sequence: true,
                  deletedAt: true,
                  case: { select: { id: true, location: true, woundType: true, patient: { select: { patientId: true, firstName: true, lastName: true } } } },
                },
              },
            },
          },
        },
      });
      if (rows.length === 0) return;
      let chunk = '';
      for (const r of rows) {
        const t = r.phase.treatment;
        if (t.deletedAt) continue;
        const label = (r.findings as { wound_type?: { label?: string } } | null)?.wound_type?.label ?? '';
        chunk += csvRow([
          t.case.patient.patientId,
          ...(deid ? [] : [`${t.case.patient.firstName} ${t.case.patient.lastName}`]),
          t.case.id,
          t.case.location,
          t.case.woundType,
          `T${t.sequence}`,
          deid ? month(r.createdAt) : dayOf(r.createdAt),
          r.area,
          r.length,
          r.height,
          label,
          r.confidenceScore,
          r.urgent,
          r.flagCount,
          r.reviewStatus,
          r.modelVersions ? JSON.stringify(r.modelVersions) : '',
        ]);
      }
      yield chunk;
      after = rows[rows.length - 1].id;
    }
  }
  /**
   * One row per analysed visit: the nurse's own assessment next to what the model found in the same (before-treatment)
   * photo, and the clinician's review. wound-ai/scripts/clinic_validation.py scores the model from it.
   * The nurse's wound type is the case's latest answer (a wound's type rarely changes between visits).
   */
  private async *validation(ctx: ClinicContext): AsyncGenerator<string> {
    yield csvRow([
      'patient_code', 'case_id', 'visit', 'visit_month', 'wound_location',
      'nurse_wound_type', 'nurse_wound_bed_tissue', 'nurse_pressure_stage', 'nurse_burn_depth', 'nurse_wagner_grade',
      'nurse_periwound', 'nurse_edge', 'nurse_infection_signs', 'nurse_exudate_level',
      'model_wound_type', 'model_wound_type_prob', 'model_severity', 'model_tissue_pct', 'model_tissue_untrusted',
      'model_periwound_erythema_frac', 'model_periwound_maceration_frac', 'model_periwound_callus_frac', 'area_cm2',
      'review', 'corrections', 'model_versions', 'flags_version',
      // Random ids, no identity: for wound-ai/scripts/export_for_labelling.py to fetch the photo with the clinic's key.
      'visit_id', 'photo_path',
    ]);
    let after: string | undefined;
    for (;;) {
      const rows = await this.prisma.aIResult.findMany({
        where: { clinicId: ctx.clinicId, status: 'ok', ...PRE_VISIT, ...(after ? { id: { gt: after } } : {}) },
        orderBy: { id: 'asc' },
        take: BATCH,
        select: {
          id: true,
          createdAt: true,
          area: true,
          reviewStatus: true,
          modelVersions: true,
          findings: true,
          review: { select: { corrections: true } },
          phase: {
            select: {
              assessment: true,
              image: { select: { imageUrl: true } },
              treatment: {
                select: {
                  sequence: true,
                  deletedAt: true,
                  case: { select: { id: true, location: true, woundType: true, patient: { select: { patientId: true } } } },
                },
              },
            },
          },
        },
      });
      if (rows.length === 0) return;
      let chunk = '';
      for (const r of rows) {
        const t = r.phase.treatment;
        if (t.deletedAt) continue;
        const a = r.phase.assessment;
        const f = (r.findings ?? {}) as ModelFindings;
        chunk += csvRow([
          t.case.patient.patientId,
          t.case.id,
          `T${t.sequence}`,
          month(r.createdAt),
          t.case.location,
          t.case.woundType,
          joined(a?.woundBedTissue),
          a?.pressureStage,
          a?.burnDepth,
          a?.wagnerGrade,
          a?.periwoundCondition,
          a?.edgeCondition,
          joined(a?.infectionSigns),
          a?.exudateLevel,
          f.wound_type?.label,
          f.wound_type?.prob,
          json(f.severity),
          json(f.tissue_pct),
          joined(f.tissue_untrusted),
          f.periwound_erythema_frac,
          f.periwound_maceration_frac,
          f.periwound_callus_frac,
          r.area,
          r.reviewStatus,
          json(r.review?.corrections),
          json(r.modelVersions),
          f.flags_version,
          r.id,
          r.phase.image?.imageUrl,
        ]);
      }
      yield chunk;
      after = rows[rows.length - 1].id;
    }
  }
}
