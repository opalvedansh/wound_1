import { Injectable } from '@nestjs/common';
import { isUncertain, woundTypeName, type AnalyzeResponse } from '@antigravity-project-spec-pack/domain/wound-model';
import type { WoundStatus } from '@antigravity-project-spec-pack/domain/api';
import { CacheService } from '../platform/cache.service';
import { PrismaService } from '../prisma.service';

const DAY_MS = 86_400_000;
/** Growth above this since the previous measured visit sends a wound to review. */
const GROWTH_REVIEW_PCT = 10;
const SEVERITY: Record<WoundStatus, number> = { healing: 0, review: 1, overdue: 2 };
const UNKNOWN_TYPES = new Set(['', 'Not recorded', 'Unknown']);

const startOfDay = (d: Date) => Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());

interface VisitFacts {
  at: Date;
  area: number | null;
  urgent: boolean;
  findings: AnalyzeResponse | null;
}

/** The status rule for one wound, from its analysed visits (oldest first) and planned next visit. */
export function woundStatus(visits: VisitFacts[], nextVisitDue: Date | null, now = new Date()): { status: WoundStatus | null; reason: string | null } {
  if (visits.length === 0) return { status: null, reason: null };
  if (nextVisitDue && startOfDay(nextVisitDue) < startOfDay(now)) {
    const days = Math.round((startOfDay(now) - startOfDay(nextVisitDue)) / DAY_MS);
    return { status: 'overdue', reason: `Visit overdue by ${days} ${days === 1 ? 'day' : 'days'}` };
  }
  const latest = visits[visits.length - 1];
  if (latest.urgent) return { status: 'review', reason: 'Urgent flag on the latest visit' };
  if (isUncertain(latest.findings?.wound_type)) return { status: 'review', reason: 'Model unsure of the wound type' };
  const measured = visits.filter((v) => v.area !== null);
  if (measured.length >= 2) {
    const [prev, last] = measured.slice(-2);
    const growth = ((last.area! - prev.area!) / prev.area!) * 100;
    if (prev.area! > 0 && growth > GROWTH_REVIEW_PCT) return { status: 'review', reason: `Wound grew ${Math.round(growth)}% since the last visit` };
  }
  return { status: 'healing', reason: null };
}

export const worst = (statuses: (WoundStatus | null)[]): WoundStatus | null =>
  statuses.reduce<WoundStatus | null>((w, s) => (s && (!w || SEVERITY[s] > SEVERITY[w]) ? s : w), null);

/**
 * Keeps the summary columns on Case and Patient in step with the visits, so lists, the review queue and the
 * dashboard read one indexed row per wound instead of scanning visits. Called after every visit result,
 * review, delete and next-visit change.
 */
@Injectable()
export class SummaryService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly cache: CacheService,
  ) {}

  async refreshCase(caseId: string): Promise<void> {
    const c = await this.prisma.case.findUnique({
      where: { id: caseId },
      select: {
        id: true,
        clinicId: true,
        patientId: true,
        woundType: true,
        treatments: {
          where: { deletedAt: null },
          orderBy: { createdAt: 'asc' },
          select: {
            nextVisit: true,
            phases: {
              where: { deletedAt: null, phaseType: 'PRE' },
              select: { aiResult: { select: { id: true, status: true, area: true, urgent: true, findings: true, createdAt: true } } },
            },
          },
        },
      },
    });
    if (!c) return;

    const results = c.treatments
      .flatMap((t) => t.phases.map((p) => p.aiResult))
      .filter((r): r is NonNullable<typeof r> => !!r && r.status === 'ok')
      .sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime());
    const visits: VisitFacts[] = results.map((r) => ({
      at: r.createdAt,
      area: r.area,
      urgent: r.urgent,
      findings: r.findings as AnalyzeResponse | null,
    }));
    const nextVisitDue = [...c.treatments].reverse().find((t) => t.nextVisit)?.nextVisit ?? null;
    const { status, reason } = woundStatus(visits, nextVisitDue);
    const measured = visits.filter((v) => v.area !== null);
    const latest = results[results.length - 1];

    // Fill in the wound type from the model only when the clinician hasn't recorded one.
    const typed = [...visits].reverse().find((v) => v.findings?.wound_type && !isUncertain(v.findings.wound_type))?.findings?.wound_type;
    const woundType = UNKNOWN_TYPES.has(c.woundType) && typed ? woundTypeName(typed.label) : undefined;

    await this.prisma.case.update({
      where: { id: caseId },
      data: {
        status,
        statusReason: reason,
        visitCount: visits.length,
        lastVisitAt: latest?.createdAt ?? null,
        nextVisitDue,
        firstAreaCm2: measured[0]?.area ?? null,
        latestAreaCm2: measured[measured.length - 1]?.area ?? null,
        latestResultId: latest?.id ?? null,
        ...(woundType ? { woundType } : {}),
      },
    });
    await this.refreshPatient(c.patientId);
    await this.cache.bump(c.clinicId);
  }

  async refreshPatient(patientId: string): Promise<void> {
    const open = await this.prisma.case.findMany({
      where: { patientId, deletedAt: null, closedAt: null },
      select: { status: true, lastVisitAt: true, nextVisitDue: true },
    });
    const times = (dates: (Date | null)[]) => dates.filter((d): d is Date => !!d).map((d) => d.getTime());
    const last = times(open.map((c) => c.lastVisitAt));
    const next = times(open.map((c) => c.nextVisitDue));
    await this.prisma.patient.update({
      where: { id: patientId },
      data: {
        status: worst(open.map((c) => c.status as WoundStatus | null)),
        lastVisitAt: last.length ? new Date(Math.max(...last)) : null,
        nextVisitDue: next.length ? new Date(Math.min(...next)) : null,
      },
    });
  }
}
