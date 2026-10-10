import type { TreatmentReportResponse } from '@antigravity-project-spec-pack/domain/wound-model';
import type { SummaryService } from '../cases/summary.service';
import type { AuditService } from '../platform/audit.service';
import type { CacheService } from '../platform/cache.service';
import type { JobsService } from '../platform/jobs.service';
import type { PrismaService } from '../prisma.service';
import type { ModelClient } from './model-client';
import { TreatmentReportService } from './treatment-report.service';

const at = (day: number) => new Date(Date.UTC(2026, 0, 1 + day, 10));

const result = (id: string, day: number, area: number, overrides: Record<string, unknown> = {}) => ({
  id,
  status: 'ok',
  findings: { status: 'ok', wound_type: { label: 'venous', prob: 0.9, top: [] }, measurement: { area_cm2: area, length_cm: 3, width_cm: 2, perimeter_cm: 8, n_regions: 1 }, flags: [] },
  intake: { diabetes: 'no', cause: 'started_on_its_own', body_location: 'lower_leg' },
  createdAt: at(day),
  reviewStatus: 'pending',
  review: null,
  ...overrides,
});

const treatment = (sequence: number, pre: unknown, post: unknown = null, assessment: unknown = null) => ({
  id: `t${sequence}`,
  sequence,
  therapy: ['Cleansing'],
  dressing: 'Foam',
  phases: [
    { phaseType: 'PRE', assessment, aiResult: pre },
    ...(post ? [{ phaseType: 'POST', assessment: null, aiResult: post }] : []),
  ],
});

const REPORT: TreatmentReportResponse = {
  progress: { session: null, healing: { phase: 'pre', n_photos: 2, comparable: true, trajectory: 'improving', basis: 'area' }, push: null, flags: [] },
  suggestions: [{ rule_id: 'M2', domain: 'M', action: 'Foam', alternatives: ['Alginate'], text: 'Consider foam.', because: ['moderate exudate'] }],
  contraindications: [{ action: 'Compression', reason: 'ABPI not recorded' }],
  checks: [],
  flags: [{ level: 'review', text: 'Blood flow not assessed.' }],
  report_markdown: '# Treatment 2 assessment (AI-assisted draft)',
  rules_version: 'care-0.1-unsigned',
};

function setup(rows: unknown[]) {
  const prisma = {
    treatment: {
      findFirst: jest.fn(async () => ({ caseId: 'c1', clinicId: 'k1', sequence: rows.length })),
      findMany: jest.fn(async () => rows),
      update: jest.fn(async () => ({})),
    },
    aIResult: { update: jest.fn(async ({ data }: { data: unknown }) => data) },
    aIReview: { deleteMany: jest.fn(async () => ({ count: 1 })) },
    $transaction: jest.fn(async (queries: Promise<unknown>[]) => Promise.all(queries)),
  };
  const model = { treatmentReport: jest.fn(async () => REPORT) };
  const jobs = { handle: jest.fn() };
  const summary = { refreshCase: jest.fn(async () => undefined) };
  const cache = { bump: jest.fn(async () => undefined) };
  const audit = { log: jest.fn(async () => undefined) };
  const service = new TreatmentReportService(
    prisma as unknown as PrismaService,
    model as unknown as ModelClient,
    jobs as unknown as JobsService,
    summary as unknown as SummaryService,
    cache as unknown as CacheService,
    audit as unknown as AuditService,
  );
  return { service, prisma, model, summary, audit };
}

describe('TreatmentReportService.run', () => {
  it("sends the wound's treatments in order, with the clinician's assessment, and writes the report onto the PRE result", async () => {
    const assessment = { exudateLevel: 'Moderate', exudateType: 'Not recorded', infectionSigns: [], painLevel: 3, edgeCondition: null, periwoundCondition: 'Healthy' };
    const { service, model, prisma, summary } = setup([
      treatment(1, result('v1', 0, 10)),
      treatment(2, result('v2', 14, 7), result('v2-post', 14, 7.5, { reviewStatus: 'included' }), assessment),
    ]);
    await service.run('t2', 'post');

    const req = (model.treatmentReport.mock.calls[0] as unknown as [{ treatments: { sequence: number; pre: { area_cm2: number; taken_at: string }; post: unknown; assessment: unknown }[]; wound_type: unknown }])[0];
    expect(req.wound_type).toEqual({ label: 'venous', prob: 0.9, top: [] });
    expect(req.treatments.map((t) => t.sequence)).toEqual([1, 2]);
    expect(req.treatments[0].pre).toMatchObject({ area_cm2: 10, taken_at: at(0).toISOString() });
    expect(req.treatments[0].post).toBeNull();
    expect(req.treatments[1].post).toMatchObject({ area_cm2: 7.5 });
    // 'Not recorded' is a placeholder, not an answer.
    expect(req.treatments[1].assessment).toMatchObject({ exudate_level: 'Moderate', exudate_type: null, periwound_condition: 'Healthy', pain_level: 3 });

    expect(prisma.aIResult.update).toHaveBeenCalledWith({
      where: { id: 'v2' },
      data: expect.objectContaining({
        draftReport: REPORT.report_markdown,
        rulesVersion: 'care-0.1-unsigned',
        care: { suggestions: REPORT.suggestions, contraindications: REPORT.contraindications, checks: [] },
        urgent: false,
        flagCount: 1,
      }),
    });
    expect(prisma.aIReview.deleteMany).not.toHaveBeenCalled();
    expect(summary.refreshCase).toHaveBeenCalledWith('c1');
  });

  it('waits for the PRE photo: the report is its draft', async () => {
    const { service, model } = setup([treatment(1, result('v1', 0, 10, { status: 'processing' }))]);
    await service.run('t1', 'post');
    expect(model.treatmentReport).not.toHaveBeenCalled();
  });

  it('a post-treatment photo added after review sends the visit back for review, and the audit log keeps the decision', async () => {
    const pre = result('v1', 0, 10, { reviewStatus: 'approved', review: { decision: 'approved' } });
    const { service, prisma, audit } = setup([treatment(1, pre, result('v1-post', 0, 11))]);
    await service.run('t1', 'post');
    expect(prisma.aIReview.deleteMany).toHaveBeenCalledWith({ where: { aiResultId: 'v1' } });
    expect(prisma.aIResult.update).toHaveBeenCalledWith(expect.objectContaining({ data: expect.objectContaining({ reviewStatus: 'pending' }) }));
    expect(audit.log).toHaveBeenCalledWith(
      expect.objectContaining({ action: 'visit.review_reopened', entityId: 'v1', details: expect.objectContaining({ previousDecision: 'approved' }) }),
    );
  });

  it('never rewrites a reviewed draft for anything but a new post-treatment photo', async () => {
    const pre = result('v1', 0, 10, { reviewStatus: 'approved', review: { decision: 'approved' } });
    const { service, model } = setup([treatment(1, pre)]);
    await service.run('t1', 'pre');
    expect(model.treatmentReport).not.toHaveBeenCalled();
  });
});
