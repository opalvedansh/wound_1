import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import {
  observation,
  type AnalyzeResponse,
  type CareAssessment,
  type IntakeAnswers,
  type Observation,
  type TreatmentInput,
} from '@antigravity-project-spec-pack/domain/wound-model';
import { SummaryService } from '../cases/summary.service';
import { AuditService } from '../platform/audit.service';
import { CacheService } from '../platform/cache.service';
import { JobsService } from '../platform/jobs.service';
import { PrismaService } from '../prisma.service';
import { ModelClient } from './model-client';

/** A result the report can use: analysed, or analysed and found unusable (a POST photo is kept either way). */
const ANALYSED = ['ok', 'retake', 'no_wound_found'];

const treatmentSelect = {
  id: true,
  sequence: true,
  therapy: true,
  dressing: true,
  phases: {
    where: { deletedAt: null },
    select: {
      phaseType: true,
      assessment: true,
      aiResult: { select: { id: true, status: true, findings: true, intake: true, createdAt: true, reviewStatus: true, review: { select: { decision: true } } } },
    },
  },
} satisfies Prisma.TreatmentSelect;

type TreatmentRow = Prisma.TreatmentGetPayload<{ select: typeof treatmentSelect }>;
type PhaseRow = TreatmentRow['phases'][number];

const recorded = (v: string | null | undefined) => (v && v !== 'Not recorded' ? v : null);

const toObservation = (phase: PhaseRow | undefined): Observation | null => {
  const r = phase?.aiResult;
  return r && ANALYSED.includes(r.status) && r.findings ? observation(r.findings as unknown as AnalyzeResponse, r.createdAt.toISOString()) : null;
};

const toAssessment = (a: PhaseRow['assessment']): CareAssessment | null =>
  a
    ? {
        exudate_level: recorded(a.exudateLevel),
        exudate_type: recorded(a.exudateType),
        infection_signs: a.infectionSigns,
        edge_condition: recorded(a.edgeCondition),
        periwound_condition: recorded(a.periwoundCondition),
        pain_level: a.painLevel,
      }
    : null;

export const toTreatmentInput = (t: TreatmentRow): TreatmentInput => {
  const pre = t.phases.find((p) => p.phaseType === 'PRE');
  return {
    sequence: t.sequence,
    pre: toObservation(pre),
    post: toObservation(t.phases.find((p) => p.phaseType === 'POST')),
    assessment: toAssessment(pre?.assessment ?? null),
    therapy: t.therapy,
    dressing: t.dressing,
  };
};

/**
 * The treatment report: healing since earlier visits, what this visit's cleaning did, and rule-based care
 * suggestions (wound-ai /treatment-report). It runs after each photo of a treatment is analysed and is written
 * onto the treatment's PRE result, which stays the one draft a clinician reviews per treatment.
 */
@Injectable()
export class TreatmentReportService implements OnModuleInit {
  private readonly logger = new Logger(TreatmentReportService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly model: ModelClient,
    private readonly jobs: JobsService,
    private readonly summary: SummaryService,
    private readonly cache: CacheService,
    private readonly audit: AuditService,
  ) {}

  onModuleInit() {
    this.jobs.handle('treatment-report', (data) => this.run(String(data['treatmentId']), data['trigger'] === 'post' ? 'post' : 'pre'));
  }

  async run(treatmentId: string, trigger: 'pre' | 'post'): Promise<void> {
    const current = await this.prisma.treatment.findFirst({
      where: { id: treatmentId, deletedAt: null },
      select: { caseId: true, clinicId: true, sequence: true },
    });
    if (!current) return;
    // This wound's treatments up to this one, oldest first: earlier visits are what "healing" compares against.
    const rows = await this.prisma.treatment.findMany({
      where: { caseId: current.caseId, deletedAt: null, sequence: { lte: current.sequence } },
      orderBy: { sequence: 'asc' },
      select: treatmentSelect,
    });
    const row = rows[rows.length - 1];
    const pre = row?.phases.find((p) => p.phaseType === 'PRE')?.aiResult;
    // The report is the PRE result's draft, so there is nothing to attach it to until that photo is analysed.
    if (!pre || pre.status !== 'ok') return;
    if (pre.review && trigger === 'pre') return; // already reviewed; only a new POST photo reopens it

    const findings = pre.findings as unknown as AnalyzeResponse;
    const report = await this.model.treatmentReport({
      wound_type: findings.wound_type ?? null,
      ...(findings.severity ? { severity: findings.severity } : {}),
      intake: (pre.intake ?? {}) as IntakeAnswers,
      treatments: rows.map(toTreatmentInput),
    });

    const { suggestions, contraindications, checks } = report;
    const reopen = !!pre.review;
    await this.prisma.$transaction([
      ...(reopen ? [this.prisma.aIReview.deleteMany({ where: { aiResultId: pre.id } })] : []),
      this.prisma.aIResult.update({
        where: { id: pre.id },
        data: {
          progress: report.progress as unknown as Prisma.InputJsonValue,
          care: { suggestions, contraindications, checks } as unknown as Prisma.InputJsonValue,
          rulesVersion: report.rules_version,
          draftReport: report.report_markdown,
          urgent: report.flags.some((f) => f.level === 'urgent'),
          flagCount: report.flags.length,
          ...(reopen ? { reviewStatus: 'pending' } : {}),
        },
      }),
      this.prisma.treatment.update({ where: { id: treatmentId }, data: { version: { increment: 1 } } }),
    ]);
    if (reopen) {
      // The decision is kept here: the draft it was made on has changed, so the clinician reviews it again.
      await this.audit.log({
        clinicId: current.clinicId,
        userId: null,
        action: 'visit.review_reopened',
        entity: 'AIResult',
        entityId: pre.id,
        details: { reason: 'post-treatment photo added', previousDecision: pre.review?.decision },
      });
    }
    await Promise.all([this.summary.refreshCase(current.caseId), this.cache.bump(current.clinicId)]);
    this.logger.log(`Treatment report for ${treatmentId} (${trigger}): ${suggestions.length} suggestion(s).`);
  }
}
