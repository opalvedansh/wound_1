import { Injectable, NotFoundException } from '@nestjs/common';
import type { Prisma } from '@prisma/client';
import { z } from 'zod';
import type { CaseCard, CaseView, Page, VisitView } from '@antigravity-project-spec-pack/domain/api';
import type { ClinicContext } from '../auth/clinic.guard';
import { AuditService } from '../platform/audit.service';
import { CacheService } from '../platform/cache.service';
import { decodeCursor, limitFrom, toPage } from '../platform/pagination';
import { calendarDay, optionalText, parse, pastDay, text } from '../platform/validation';
import { PrismaService } from '../prisma.service';
import { StorageService } from '../visits/storage.service';
import { toCaseCard, toVisitView, visitPaths, visitSelect } from '../views';
import { SummaryService } from './summary.service';

export const createCase = z.object({
  patientId: z.string().uuid(),
  location: text(200),
  onset: pastDay,
  woundType: z.string().trim().max(100).optional(),
  comorbidities: z.array(z.string().trim().min(1).max(100)).max(30).default([]),
  remarks: optionalText(2000),
});

export const updateCase = z.object({
  location: text(200).optional(),
  woundType: z.string().trim().max(100).optional(),
  comorbidities: z.array(z.string().trim().min(1).max(100)).max(30).optional(),
  remarks: z.string().trim().max(2000).nullable().optional(),
  closed: z.boolean().optional(),
  /** Planned next visit (YYYY-MM-DD) on the latest visit; null clears it. */
  nextVisit: calendarDay.nullable().optional(),
});

const VISITS_PAGE = 10;

@Injectable()
export class CasesService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
    private readonly summary: SummaryService,
    private readonly cache: CacheService,
    private readonly audit: AuditService,
  ) {}

  private async owned(ctx: ClinicContext, id: string) {
    const c = await this.prisma.case.findFirst({
      where: { id, clinicId: ctx.clinicId, deletedAt: null },
      include: { patient: { select: { id: true, firstName: true, lastName: true, patientId: true } } },
    });
    if (!c) throw new NotFoundException('Wound not found.');
    return c;
  }

  async get(ctx: ClinicContext, id: string): Promise<CaseView> {
    // All three are clinic-scoped, so they run together; nothing is returned unless the wound is found.
    const [c, visits, series] = await Promise.all([
      this.owned(ctx, id),
      this.visits(ctx, id, {}),
      this.prisma.aIResult.findMany({
        where: { clinicId: ctx.clinicId, status: 'ok', area: { not: null }, phase: { phaseType: 'PRE', treatment: { caseId: id, deletedAt: null } } },
        orderBy: { createdAt: 'asc' },
        select: { area: true, createdAt: true, phase: { select: { treatment: { select: { sequence: true } } } } },
      }),
    ]);
    const latest = visits.items.find((v) => v.status === 'ok');
    const card: CaseCard = toCaseCard(c, latest?.thumbUrl ?? latest?.photoUrl ?? null, latest?.findings?.outline ?? null);
    return {
      ...card,
      comorbidities: c.comorbidities,
      remarks: c.remarks,
      patient: c.patient,
      areaSeries: series.map((s) => ({ at: s.createdAt.toISOString(), areaCm2: s.area as number, sequence: s.phase.treatment.sequence })),
      visits,
    };
  }

  /** Newest first, keyset-paginated. Only analysed or in-progress visits; retakes are not kept. */
  async visits(ctx: ClinicContext, caseId: string, query: { cursor?: unknown; limit?: unknown }): Promise<Page<VisitView>> {
    const limit = query.limit === undefined ? VISITS_PAGE : limitFrom(query.limit);
    const cursor = decodeCursor(query.cursor);
    const after: Prisma.AIResultWhereInput = cursor
      ? { OR: [{ createdAt: { lt: new Date(String(cursor.v)) } }, { createdAt: new Date(String(cursor.v)), id: { lt: cursor.id } }] }
      : {};
    const rows = await this.prisma.aIResult.findMany({
      where: {
        AND: [
          { clinicId: ctx.clinicId, status: { in: ['ok', 'processing', 'failed'] }, phase: { phaseType: 'PRE', treatment: { caseId, deletedAt: null } } },
          after,
        ],
      },
      orderBy: [{ createdAt: 'desc' }, { id: 'desc' }],
      take: limit + 1,
      select: visitSelect,
    });
    const page = toPage(rows, limit, (r) => r.createdAt.toISOString());
    const urls = await this.storage.signedUrls(visitPaths(page.items));
    return { items: page.items.map((r) => toVisitView(r, urls)), nextCursor: page.nextCursor };
  }

  async create(ctx: ClinicContext, raw: unknown) {
    const input = parse(createCase, raw, 'Check the wound details.');
    const patient = await this.prisma.patient.findFirst({ where: { id: input.patientId, clinicId: ctx.clinicId, deletedAt: null }, select: { id: true } });
    if (!patient) throw new NotFoundException('Patient not found.');
    const created = await this.prisma.case.create({
      data: {
        clinicId: ctx.clinicId,
        patientId: input.patientId,
        location: input.location,
        onset: input.onset,
        woundType: input.woundType || 'Not recorded',
        comorbidities: input.comorbidities,
        remarks: input.remarks,
      },
      select: { id: true },
    });
    await Promise.all([
      this.summary.refreshPatient(input.patientId),
      this.cache.bump(ctx.clinicId),
      this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'case.create', entity: 'Case', entityId: created.id }),
    ]);
    return this.get(ctx, created.id);
  }

  async update(ctx: ClinicContext, id: string, raw: unknown) {
    const input = parse(updateCase, raw);
    await this.owned(ctx, id);
    const { closed, nextVisit, ...fields } = input;
    await this.prisma.case.update({
      where: { id },
      data: {
        ...fields,
        ...(closed === undefined ? {} : { closedAt: closed ? new Date() : null }),
        version: { increment: 1 },
      },
    });
    if (nextVisit !== undefined) {
      const latest = await this.prisma.treatment.findFirst({ where: { caseId: id, deletedAt: null }, orderBy: { createdAt: 'desc' }, select: { id: true } });
      if (latest) await this.prisma.treatment.update({ where: { id: latest.id }, data: { nextVisit, version: { increment: 1 } } });
    }
    await this.summary.refreshCase(id);
    await this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'case.update', entity: 'Case', entityId: id, details: { fields: Object.keys(input) } });
    return this.get(ctx, id);
  }

  /** Marks a wound reviewed (it leaves the review queue until its next visit), or undoes that. */
  async markReviewed(ctx: ClinicContext, id: string, reviewed: boolean) {
    await this.owned(ctx, id);
    const updated = await this.prisma.case.update({
      where: { id },
      data: { reviewedAt: reviewed ? new Date() : null, version: { increment: 1 } },
    });
    await Promise.all([
      this.cache.bump(ctx.clinicId),
      this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: reviewed ? 'case.reviewed' : 'case.unreviewed', entity: 'Case', entityId: id }),
    ]);
    return toCaseCard(updated);
  }
}
