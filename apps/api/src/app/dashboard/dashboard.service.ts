import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import type {
  AttentionItem,
  Dashboard,
  DraftItem,
  Page,
  QueueCounts,
  QueueView,
  WoundStatus,
} from '@antigravity-project-spec-pack/domain/api';
import type { ClinicContext } from '../auth/clinic.guard';
import { CacheService } from '../platform/cache.service';
import { decodeCursor, encodeCursor, limitFrom } from '../platform/pagination';
import { PatientsService } from '../patients/patients.service';
import { PrismaService } from '../prisma.service';
import { StorageService } from '../visits/storage.service';
import { areaChangePct, day, iso, PRE_VISIT } from '../views';

const DAY_MS = 86_400_000;
const DASHBOARD_TTL = 30;
// "Not reviewed since its last visit" compares two columns, which Prisma's query builder can't express.
// Qualified with the Case table's alias, because Patient has a lastVisitAt column too.
const unreviewed = (t = 'c') =>
  Prisma.sql`(${Prisma.raw(t)}."reviewedAt" IS NULL OR (${Prisma.raw(t)}."lastVisitAt" IS NOT NULL AND ${Prisma.raw(t)}."reviewedAt" < ${Prisma.raw(t)}."lastVisitAt"))`;

const name = (p: { firstName: string; lastName: string }) => `${p.firstName} ${p.lastName}`.trim();

interface AttentionRow {
  id: string;
  patientId: string;
  firstName: string;
  lastName: string;
  location: string;
  status: WoundStatus;
  statusReason: string | null;
  lastVisitAt: Date | null;
  nextVisitDue: Date | null;
  reviewedAt: Date | null;
  latestResultId: string | null;
}

/**
 * The dashboard and the review queue. Everything is computed from the summary columns on Case (kept current by
 * SummaryService) with indexed, clinic-scoped queries, and the dashboard is cached per clinic until the next write.
 */
@Injectable()
export class DashboardService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly cache: CacheService,
    private readonly storage: StorageService,
    private readonly patients: PatientsService,
  ) {}

  dashboard(ctx: ClinicContext): Promise<Dashboard> {
    return this.cache.clinic(ctx.clinicId, 'dashboard', DASHBOARD_TTL, () => this.computeDashboard(ctx));
  }

  private async computeDashboard(ctx: ClinicContext): Promise<Dashboard> {
    const clinicId = ctx.clinicId;
    const now = new Date();
    const weekAgo = new Date(now.getTime() - 7 * DAY_MS);
    const twoWeeksAgo = new Date(now.getTime() - 14 * DAY_MS);
    const eightWeeksAgo = new Date(now.getTime() - 56 * DAY_MS);
    const inAWeek = new Date(now.getTime() + 7 * DAY_MS);
    const openCase = { clinicId, deletedAt: null, closedAt: null };
    const okVisit = { clinicId, status: 'ok', ...PRE_VISIT };

    const [patients, openWounds, thisWeek, lastWeek, drafts, urgentDrafts, attentionCount, weekly, types, progress, attention, upcoming] =
      await Promise.all([
        this.prisma.patient.count({ where: { clinicId, deletedAt: null } }),
        this.prisma.case.count({ where: openCase }),
        this.prisma.aIResult.count({ where: { ...okVisit, createdAt: { gte: weekAgo } } }),
        this.prisma.aIResult.count({ where: { ...okVisit, createdAt: { gte: twoWeeksAgo, lt: weekAgo } } }),
        this.prisma.aIResult.count({ where: { ...okVisit, reviewStatus: 'pending' } }),
        this.prisma.aIResult.count({ where: { ...okVisit, reviewStatus: 'pending', urgent: true } }),
        this.prisma.$queryRaw<{ n: bigint }[]>`
          SELECT count(*) AS n FROM "Case" c
          WHERE c."clinicId" = ${clinicId} AND c."deletedAt" IS NULL AND c."closedAt" IS NULL
            AND c.status IN ('overdue', 'review') AND ${unreviewed()}`,
        this.prisma.$queryRaw<{ week: Date; n: bigint }[]>`
          SELECT date_trunc('week', "createdAt") AS week, count(*) AS n FROM "AIResult"
          WHERE "clinicId" = ${clinicId} AND status = 'ok' AND "reviewStatus" <> 'included' AND "createdAt" >= ${eightWeeksAgo}
          GROUP BY 1 ORDER BY 1`,
        this.prisma.case.groupBy({ by: ['woundType'], where: openCase, _count: { _all: true } }),
        this.prisma.case.findMany({
          where: { ...openCase, firstAreaCm2: { gt: 0 }, latestAreaCm2: { not: null }, visitCount: { gte: 2 } },
          select: { id: true, location: true, firstAreaCm2: true, latestAreaCm2: true, patient: { select: { firstName: true, lastName: true } } },
          orderBy: { lastVisitAt: 'desc' },
          take: 40,
        }),
        this.attentionRows(clinicId, ['overdue', 'review'], false, 0, 5),
        this.prisma.case.findMany({
          where: { ...openCase, nextVisitDue: { gte: new Date(now.toISOString().slice(0, 10)), lte: inAWeek } },
          orderBy: { nextVisitDue: 'asc' },
          take: 6,
          select: { id: true, location: true, nextVisitDue: true, patient: { select: { id: true, firstName: true, lastName: true } } },
        }),
      ]);

    // Eight weekly buckets, including empty weeks.
    const byWeek = new Map(weekly.map((w) => [day(w.week), Number(w.n)]));
    const monday = (d: Date) => {
      const x = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
      x.setUTCDate(x.getUTCDate() - ((x.getUTCDay() + 6) % 7));
      return x;
    };
    const visitsPerWeek = Array.from({ length: 8 }, (_, i) => {
      const start = monday(new Date(now.getTime() - (7 - i) * 7 * DAY_MS));
      const key = day(start) as string;
      return { weekStart: key, count: byWeek.get(key) ?? 0 };
    });

    const healing = progress
      .map((c) => ({ caseId: c.id, patient: name(c.patient), location: c.location, changePct: areaChangePct(c.firstAreaCm2, c.latestAreaCm2) as number }))
      .filter((h) => h.changePct !== null)
      .sort((a, b) => a.changePct - b.changePct)
      .slice(0, 12);

    return {
      patients,
      openWounds,
      shrinking: progress.filter((c) => (c.latestAreaCm2 ?? 0) < (c.firstAreaCm2 ?? 0)).length,
      visitsThisWeek: thisWeek,
      visitsLastWeek: lastWeek,
      draftsToReview: drafts,
      urgentDrafts,
      needsAttention: Number(attentionCount[0]?.n ?? 0),
      visitsPerWeek,
      woundTypes: types.map((t) => ({ label: t.woundType, count: t._count._all })).sort((a, b) => b.count - a.count),
      healing,
      attention: await this.toAttention(attention),
      upcoming: upcoming.map((u) => ({
        caseId: u.id,
        patientId: u.patient.id,
        patient: name(u.patient),
        location: u.location,
        due: day(u.nextVisitDue) as string,
      })),
    };
  }

  private attentionRows(clinicId: string, statuses: WoundStatus[], reviewed: boolean, offset: number, limit: number) {
    return this.prisma.$queryRaw<AttentionRow[]>`
      SELECT c.id, c."patientId", p."firstName", p."lastName", c.location, c.status, c."statusReason",
             c."lastVisitAt", c."nextVisitDue", c."reviewedAt", c."latestResultId"
      FROM "Case" c JOIN "Patient" p ON p.id = c."patientId"
      WHERE c."clinicId" = ${clinicId} AND c."deletedAt" IS NULL AND c."closedAt" IS NULL
        AND c.status IN (${Prisma.join(statuses)})
        AND ${reviewed ? Prisma.sql`NOT ${unreviewed()}` : unreviewed()}
      ORDER BY (c.status = 'overdue') DESC, c."nextVisitDue" ASC NULLS LAST, c."lastVisitAt" DESC
      OFFSET ${offset} LIMIT ${limit}`;
  }

  private async toAttention(rows: AttentionRow[]): Promise<AttentionItem[]> {
    const photos = await this.patients.latestPhotos(rows.map((r) => r.latestResultId).filter((x): x is string => !!x));
    return rows.map((r) => ({
      caseId: r.id,
      patientId: r.patientId,
      patient: name(r),
      location: r.location,
      status: r.status,
      reason: r.statusReason,
      lastVisitAt: iso(r.lastVisitAt),
      nextVisitDue: day(r.nextVisitDue),
      reviewedAt: iso(r.reviewedAt),
      thumbUrl: r.latestResultId ? photos.get(r.latestResultId)?.url ?? null : null,
    }));
  }

  /** Counts for the queue's filter chips (and the sidebar badge), cached until the next write. */
  counts(ctx: ClinicContext): Promise<QueueCounts> {
    return this.cache.clinic(ctx.clinicId, 'queue-counts', DASHBOARD_TTL, async () => {
      const [drafts, rows] = await Promise.all([
        this.prisma.aIResult.count({ where: { clinicId: ctx.clinicId, status: 'ok', reviewStatus: 'pending' } }),
        this.prisma.$queryRaw<{ status: string; reviewed: boolean; n: bigint }[]>`
          SELECT c.status, NOT ${unreviewed()} AS reviewed, count(*) AS n FROM "Case" c
          WHERE c."clinicId" = ${ctx.clinicId} AND c."deletedAt" IS NULL AND c."closedAt" IS NULL AND c.status IN ('overdue', 'review')
          GROUP BY 1, 2`,
      ]);
      const n = (status: string, reviewed: boolean) => Number(rows.find((r) => r.status === status && r.reviewed === reviewed)?.n ?? 0);
      return {
        drafts,
        attention: n('overdue', false) + n('review', false),
        overdue: n('overdue', false),
        review: n('review', false),
        reviewed: n('overdue', true) + n('review', true),
      };
    });
  }

  /** One page of the queue. Drafts are AI results awaiting a clinician; the other views are wounds. */
  async queue(ctx: ClinicContext, view: QueueView, cursorRaw: unknown, limitRaw: unknown): Promise<Page<DraftItem | AttentionItem>> {
    // The first page of each view is cached until the next write in the clinic.
    if (!cursorRaw) return this.cache.clinic(ctx.clinicId, `queue:${view}:${limitFrom(limitRaw)}`, 30, () => this.queuePage(ctx, view, null, limitRaw));
    return this.queuePage(ctx, view, cursorRaw, limitRaw);
  }

  private async queuePage(ctx: ClinicContext, view: QueueView, cursorRaw: unknown, limitRaw: unknown): Promise<Page<DraftItem | AttentionItem>> {
    const limit = limitFrom(limitRaw);
    const offset = Number(decodeCursor(cursorRaw)?.v ?? 0) || 0;
    const next = (count: number) => (count > limit ? encodeCursor({ v: offset + limit, id: 'offset' }) : null);

    if (view === 'drafts') {
      const rows = await this.prisma.aIResult.findMany({
        where: { clinicId: ctx.clinicId, status: 'ok', reviewStatus: 'pending' },
        orderBy: [{ urgent: 'desc' }, { createdAt: 'desc' }],
        skip: offset,
        take: limit + 1,
        select: {
          id: true,
          createdAt: true,
          urgent: true,
          flagCount: true,
          area: true,
          phase: {
            select: {
              image: { select: { thumbPath: true, imageUrl: true } },
              treatment: { select: { case: { select: { id: true, location: true, patient: { select: { firstName: true, lastName: true } } } } } },
            },
          },
        },
      });
      const page = rows.slice(0, limit);
      const urls = await this.storage.signedUrls(page.map((r) => r.phase.image?.thumbPath ?? r.phase.image?.imageUrl ?? ''));
      return {
        items: page.map((r) => {
          const c = r.phase.treatment.case;
          const path = r.phase.image?.thumbPath ?? r.phase.image?.imageUrl;
          return {
            visitId: r.id,
            caseId: c.id,
            patient: name(c.patient),
            location: c.location,
            takenAt: r.createdAt.toISOString(),
            urgent: r.urgent,
            flagCount: r.flagCount,
            areaCm2: r.area,
            thumbUrl: path ? urls.get(path) ?? null : null,
          };
        }),
        nextCursor: next(rows.length),
      };
    }

    const statuses: WoundStatus[] = view === 'overdue' ? ['overdue'] : view === 'review' ? ['review'] : ['overdue', 'review'];
    const rows = await this.attentionRows(ctx.clinicId, statuses, view === 'reviewed', offset, limit + 1);
    return { items: await this.toAttention(rows.slice(0, limit)), nextCursor: next(rows.length) };
  }
}
