import { BadRequestException, ConflictException, NotFoundException, PayloadTooLargeException } from '@nestjs/common';
import type { AnalyzeResponse, IntakeQuestion } from '@antigravity-project-spec-pack/domain/wound-model';
import type { ClinicContext } from '../auth/clinic.guard';
import type { SummaryService } from '../cases/summary.service';
import type { AuditService } from '../platform/audit.service';
import type { CacheService } from '../platform/cache.service';
import type { JobsService } from '../platform/jobs.service';
import type { PrismaService } from '../prisma.service';
import type { UsersService } from '../users.service';
import type { ModelClient } from './model-client';
import type { StorageService } from './storage.service';
import { MAX_PHOTO_BYTES, VisitsService } from './visits.service';

const ctx: ClinicContext = { userId: 'u1', email: 'dr@clinic.example', clinicId: 'k1', role: 'DOCTOR' };
const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
const photo = { buffer: jpeg, size: jpeg.length };

const CORE: IntakeQuestion[] = [
  { id: 'body_location', text: 'Where?', type: 'choice', options: ['heel', 'toe'] },
  { id: 'cause', text: 'How?', type: 'choice', options: ['burn', 'unknown'] },
  { id: 'diabetes', text: 'Diabetes?', type: 'choice', options: ['yes', 'no'] },
  { id: 'pain', text: 'Pain?', type: 'number' },
  { id: 'current_treatment', text: 'Current dressing (free text)', type: 'text' },
];
const FOLLOW_UPS: IntakeQuestion[] = [{ id: 'burn_agent', text: 'Burn agent?', type: 'choice', options: ['flame', 'chemical'] }];

const ANSWERS = {
  diabetes: 'no',
  cause: 'burn',
  body_location: 'heel',
  pain: '4',
  burn_agent: 'chemical',
  current_treatment: "Foam dressing, Mrs Rao's",
  patient_name: 'Asha Rao',
};

const OK: AnalyzeResponse = {
  status: 'ok',
  case_id: 'model-case-1',
  flags: [{ level: 'urgent', text: 'Spreading redness.' }],
  wound_type: { label: 'burn', prob: 0.86, top: [['burn', 0.86]] },
  measurement: { area_cm2: 4.2, length_cm: 3.1, width_cm: 1.8, perimeter_cm: 8.4, n_regions: 1 },
  report_markdown: '# Wound assessment (AI-assisted draft)',
  model_versions: { boundary: 'b1' },
};

const DAY = 86_400_000;

/** A stored visit as `visitSelect` returns it. */
const visitRow = (id: string, status = 'processing') => ({
  id,
  status,
  error: null,
  findings: null,
  intake: {},
  draftReport: null,
  createdAt: new Date('2026-10-08T10:00:00Z'),
  review: null,
  progress: null,
  care: null,
  rulesVersion: null,
  phase: { treatment: { id: 't1', sequence: 1, phases: [] }, image: { imageUrl: 'k1/treatments/t1/pre.jpg', thumbPath: null } },
});

/** The row the analyze job loads. */
const pending = (overrides: Record<string, unknown> = {}) => ({
  id: 'v1',
  status: 'processing',
  intake: { diabetes: 'no', cause: 'burn' },
  phase: {
    id: 'ph1',
    phaseType: 'PRE',
    treatmentId: 't1',
    image: { imageUrl: 'k1/treatments/t1/pre.jpg' },
    treatment: { caseId: 'c1', case: { latestAreaCm2: 5, lastVisitAt: new Date(Date.now() - 7 * DAY) } },
  },
  ...overrides,
});

function setup(
  options: { transactionFails?: boolean; result?: unknown; job?: unknown; findings?: AnalyzeResponse; modelFails?: boolean; repeat?: boolean; treatment?: unknown } = {},
) {
  const created: Record<string, Record<string, unknown>[]> = { treatment: [], phase: [], image: [], aIResult: [] };
  const record = (table: string) => ({
    create: jest.fn(async ({ data }: { data: Record<string, unknown> }) => {
      if (table === 'aIResult' && options.transactionFails) throw new Error('database unavailable');
      const row = { id: table === 'aIResult' ? 'v1' : `${table}-1`, ...data };
      created[table].push(row);
      return row;
    }),
    findFirst: jest.fn(async () => null),
  });
  const tx = { treatment: record('treatment'), phase: record('phase'), image: record('image'), aIResult: record('aIResult') };
  const prisma = {
    case: {
      findFirst: jest.fn(async ({ where }: { where: { id: string; clinicId: string } }) =>
        where.id === 'c1' && where.clinicId === 'k1' ? { id: 'c1', patientId: 'p1' } : null,
      ),
    },
    // Interactive transactions get the fake `tx`; batched ones are a list of queries.
    $transaction: jest.fn(async (arg: ((t: typeof tx) => unknown) | Promise<unknown>[]) => (typeof arg === 'function' ? arg(tx) : Promise.all(arg))),
    aIResult: {
      findFirst: jest.fn(async ({ where }: { where: { id: string; clinicId: string } }) =>
        where.clinicId !== 'k1' ? null : options.result === undefined ? visitRow(where.id) : options.result,
      ),
      findUnique: jest.fn(async () => options.job ?? null),
      update: jest.fn(async ({ data }: { data: Record<string, unknown> }) => data),
    },
    aIReview: { create: jest.fn(async ({ data }: { data: Record<string, unknown> }) => ({ ...data, createdAt: new Date('2026-10-08T11:00:00Z') })) },
    image: { deleteMany: jest.fn(async () => ({ count: 1 })), findFirst: jest.fn(async () => (options.repeat ? { phaseId: 'old' } : null)) },
    treatment: { updateMany: jest.fn(async () => ({ count: 1 })), findFirst: jest.fn(async () => options.treatment ?? null) },
  };
  const model = {
    questions: jest.fn(async () => CORE),
    followUps: jest.fn(async () => FOLLOW_UPS),
    check: jest.fn(async () => ({ quality: { ok: true, usable: true, issues: [] }, marker_found: false, phone_reading: null })),
    analyze: jest.fn(async () => {
      if (options.modelFails) throw new Error('model asleep');
      return options.findings ?? OK;
    }),
    review: jest.fn(async () => undefined),
  };
  const storage = {
    upload: jest.fn(async () => undefined),
    download: jest.fn(async () => jpeg),
    remove: jest.fn(async () => undefined),
    signedUrls: jest.fn(async (paths: string[]) => new Map(paths.map((p) => [p, `https://signed.example/${p}`]))),
  };
  const users = { ensure: jest.fn(async () => undefined) };
  const jobs = { enqueue: jest.fn(async () => undefined), handle: jest.fn() };
  const summary = { refreshCase: jest.fn(async () => undefined) };
  const cache = { bump: jest.fn(async () => undefined) };
  const audit = { log: jest.fn(async () => undefined) };
  const service = new VisitsService(
    prisma as unknown as PrismaService,
    model as unknown as ModelClient,
    storage as unknown as StorageService,
    users as unknown as UsersService,
    jobs as unknown as JobsService,
    summary as unknown as SummaryService,
    cache as unknown as CacheService,
    audit as unknown as AuditService,
  );
  return { service, prisma, model, storage, users, jobs, summary, cache, audit, created };
}

describe('VisitsService.create', () => {
  it('stores the photo under the clinic, records a processing visit and queues the analysis', async () => {
    const { service, storage, created, jobs, model } = setup();
    const visit = await service.create(ctx, 'c1', photo, JSON.stringify(ANSWERS));

    const [path, body, type] = storage.upload.mock.calls[0] as unknown as [string, Buffer, string];
    const treatmentId = created['treatment'][0]['id'] as string;
    expect(path).toBe(`k1/treatments/${treatmentId}/pre.jpg`);
    expect(body).toBe(jpeg);
    expect(type).toBe('image/jpeg');
    expect(created['treatment'][0]).toEqual(expect.objectContaining({ clinicId: 'k1', caseId: 'c1', sequence: 1 }));
    expect(created['image'][0]).toEqual(expect.objectContaining({ imageUrl: path }));
    expect(created['aIResult'][0]).toEqual(expect.objectContaining({ clinicId: 'k1', status: 'processing' }));
    expect(jobs.enqueue).toHaveBeenCalledWith('analyze-visit', { visitId: 'v1' }, { jobId: 'analyze-v1' });
    expect(model.analyze).not.toHaveBeenCalled(); // never inside the request
    expect(visit).toEqual(expect.objectContaining({ id: 'v1', status: 'processing' }));
  });

  it("keeps only answers to the model's own questions", async () => {
    const { service, model, created } = setup();
    await service.create(ctx, 'c1', photo, JSON.stringify(ANSWERS));
    expect(model.followUps).toHaveBeenCalledWith({ diabetes: 'no', cause: 'burn', body_location: 'heel', pain: 4 });
    expect(created['aIResult'][0]['intake']).toEqual({ body_location: 'heel', cause: 'burn', diabetes: 'no', pain: 4, burn_agent: 'chemical' });
  });

  it('refuses before storing anything when something is missing or wrong', async () => {
    const { service, storage } = setup();
    await expect(service.create(ctx, 'c1', photo, JSON.stringify({ diabetes: 'no' }))).rejects.toBeInstanceOf(BadRequestException);
    await expect(service.create(ctx, 'c1', undefined, JSON.stringify(ANSWERS))).rejects.toBeInstanceOf(BadRequestException);
    const text = Buffer.from('not an image');
    await expect(service.create(ctx, 'c1', { buffer: text, size: text.length }, JSON.stringify(ANSWERS))).rejects.toBeInstanceOf(BadRequestException);
    await expect(service.create(ctx, 'c1', { buffer: jpeg, size: MAX_PHOTO_BYTES + 1 }, JSON.stringify(ANSWERS))).rejects.toBeInstanceOf(
      PayloadTooLargeException,
    );
    await expect(service.create(ctx, 'c1', photo, '[1, 2]')).rejects.toBeInstanceOf(BadRequestException);
    expect(storage.upload).not.toHaveBeenCalled();
  });

  it('refuses a photo already on this wound, before storing anything', async () => {
    const { service, storage } = setup({ repeat: true });
    await expect(service.create(ctx, 'c1', photo, JSON.stringify(ANSWERS))).rejects.toThrow('already on this wound');
    expect(storage.upload).not.toHaveBeenCalled();
  });

  it("treats another clinic's wound as not found", async () => {
    const { service, storage } = setup();
    await expect(service.create({ ...ctx, clinicId: 'k2' }, 'c1', photo, JSON.stringify(ANSWERS))).rejects.toBeInstanceOf(NotFoundException);
    expect(storage.upload).not.toHaveBeenCalled();
  });

  it('removes the stored photo when the database write fails', async () => {
    const { service, storage, jobs } = setup({ transactionFails: true });
    await expect(service.create(ctx, 'c1', photo, JSON.stringify(ANSWERS))).rejects.toThrow('database unavailable');
    const [path] = storage.upload.mock.calls[0] as unknown as [string];
    expect(storage.remove).toHaveBeenCalledWith(path);
    expect(jobs.enqueue).not.toHaveBeenCalled();
  });
});

describe('VisitsService.photoCheck', () => {
  it('sends a real image to the model and refuses anything else', async () => {
    const { service, model } = setup();
    await expect(service.photoCheck({ buffer: jpeg, size: jpeg.length })).resolves.toEqual({ quality: { ok: true, usable: true, issues: [] }, marker_found: false, phone_reading: null });
    expect(model.check).toHaveBeenCalledWith({ buffer: jpeg, mimetype: 'image/jpeg' });
    await expect(service.photoCheck(undefined)).rejects.toThrow('Add a photo');
    await expect(service.photoCheck({ buffer: Buffer.from('not an image'), size: 12 })).rejects.toThrow('JPEG or PNG');
    expect(model.check).toHaveBeenCalledTimes(1);
  });
});

describe('VisitsService.analyze (background job)', () => {
  it("writes the model's findings, refreshes the wound's summary and queues a thumbnail", async () => {
    const { service, prisma, model, summary, jobs } = setup({ job: pending() });
    await service.analyze('v1');
    expect(model.analyze).toHaveBeenCalledWith({ buffer: jpeg, mimetype: 'image/jpeg' }, { diabetes: 'no', cause: 'burn' }, { area_cm2: 5, days_ago: 7 }, 'pre');
    expect(prisma.aIResult.update).toHaveBeenCalledWith({
      where: { id: 'v1' },
      data: expect.objectContaining({ status: 'ok', area: 4.2, length: 3.1, height: 1.8, confidenceScore: 0.86, modelCaseId: 'model-case-1', urgent: true, flagCount: 1 }),
    });
    expect(summary.refreshCase).toHaveBeenCalledWith('c1');
    expect(jobs.enqueue).toHaveBeenCalledWith('thumbnail', { visitId: 'v1' });
    expect(jobs.enqueue).toHaveBeenCalledWith('treatment-report', { treatmentId: 't1', trigger: 'pre' }, expect.anything());
  });

  it('analyses a post-treatment photo as part of its visit, without a previous-area comparison', async () => {
    const job = pending({ phase: { ...pending().phase, phaseType: 'POST', image: { imageUrl: 'k1/treatments/t1/post.jpg' } } });
    const { service, model, summary, jobs } = setup({ job });
    await service.analyze('v1');
    expect(model.analyze).toHaveBeenCalledWith(expect.anything(), expect.anything(), undefined, 'post');
    expect(summary.refreshCase).not.toHaveBeenCalled(); // the wound's summary follows PRE photos
    expect(jobs.enqueue).toHaveBeenCalledWith('treatment-report', { treatmentId: 't1', trigger: 'post' }, expect.anything());
  });

  it('keeps a post-treatment photo the model cannot use: the dressing is on, so it cannot be retaken', async () => {
    const job = pending({ phase: { ...pending().phase, phaseType: 'POST', image: { imageUrl: 'k1/treatments/t1/post.jpg' } } });
    const { service, prisma, storage, jobs } = setup({ job, findings: { status: 'retake', quality: { ok: false, issues: ['Glare.'] } } });
    await service.analyze('v1');
    expect(prisma.aIResult.update).toHaveBeenCalledWith(expect.objectContaining({ data: expect.objectContaining({ status: 'retake' }) }));
    expect(prisma.image.deleteMany).not.toHaveBeenCalled();
    expect(storage.remove).not.toHaveBeenCalled();
    expect(jobs.enqueue).toHaveBeenCalledWith('treatment-report', { treatmentId: 't1', trigger: 'post' }, expect.anything());
  });

  it('never keeps a photo the model asks to retake', async () => {
    const findings: AnalyzeResponse = { status: 'retake', quality: { ok: false, issues: ['Photo looks blurry.'] } };
    const { service, prisma, storage, summary } = setup({ job: pending(), findings });
    await service.analyze('v1');
    expect(prisma.aIResult.update).toHaveBeenCalledWith(expect.objectContaining({ data: expect.objectContaining({ status: 'retake' }) }));
    expect(prisma.image.deleteMany).toHaveBeenCalledWith({ where: { phaseId: 'ph1' } });
    expect(storage.remove).toHaveBeenCalledWith('k1/treatments/t1/pre.jpg');
    expect(summary.refreshCase).not.toHaveBeenCalled();
  });

  it('lets the queue retry a model failure, and marks the visit failed after the last attempt', async () => {
    const early = setup({ job: pending(), modelFails: true });
    await expect(early.service.analyze('v1', { final: false })).rejects.toThrow('model asleep');
    expect(early.prisma.aIResult.update).toHaveBeenCalledWith({ where: { id: 'v1' }, data: { error: 'model asleep' } });

    const last = setup({ job: pending(), modelFails: true });
    await expect(last.service.analyze('v1', { final: true })).resolves.toBeUndefined();
    expect(last.prisma.aIResult.update).toHaveBeenCalledWith({ where: { id: 'v1' }, data: { error: 'model asleep', status: 'failed' } });
  });

  it('skips a visit that is no longer processing (a duplicate job)', async () => {
    const { service, model } = setup({ job: pending({ status: 'ok' }) });
    await service.analyze('v1');
    expect(model.analyze).not.toHaveBeenCalled();
  });
});

describe('VisitsService.attachSyncedPhoto', () => {
  const treatment = { id: 't1', caseId: 'c1', phases: [] };

  it('analyses the post-treatment photo too, as part of the visit rather than a draft of its own', async () => {
    const { service, created, jobs } = setup({ treatment });
    const out = await service.attachSyncedPhoto(ctx, 't1', 'POST', photo, { diabetes: 'no', cause: 'burn' });
    expect(out.visitId).toBe('v1');
    expect(created['aIResult'][0]).toMatchObject({ status: 'processing', reviewStatus: 'included' });
    expect(created['image'][0]['sha256']).toMatch(/^[0-9a-f]{64}$/);
    expect(jobs.enqueue).toHaveBeenCalledWith('analyze-visit', { visitId: 'v1' }, expect.anything());
  });

  it("keeps a photo already on this wound (last visit's, carried over) but never analyses it", async () => {
    const { service, created, jobs, audit } = setup({ treatment, repeat: true });
    expect(await service.attachSyncedPhoto(ctx, 't1', 'PRE', photo, { diabetes: 'no', cause: 'burn' })).toEqual({ visitId: null });
    expect(created['image']).toHaveLength(1);
    expect(created['aIResult']).toHaveLength(0);
    expect(jobs.enqueue).not.toHaveBeenCalled();
    expect(audit.log).toHaveBeenCalledWith(expect.objectContaining({ action: 'visit.repeat_photo' }));
  });
});

describe('VisitsService.retry', () => {
  it('only a failed analysis can be retried', async () => {
    const { service, jobs } = setup({ result: { status: 'ok' } });
    await expect(service.retry(ctx, 'v1')).rejects.toBeInstanceOf(ConflictException);
    expect(jobs.enqueue).not.toHaveBeenCalled();
  });
});

describe('VisitsService.review', () => {
  const draft = { id: 'v1', draftReport: 'Draft text', review: null };

  it('approving keeps the draft as the final report and tells the model in the background', async () => {
    const { service, prisma, jobs, cache, audit, users } = setup({ result: draft });
    const review = await service.review(ctx, 'v1', { decision: 'approved' });
    expect(users.ensure).toHaveBeenCalledWith({ id: 'u1', email: 'dr@clinic.example', role: null });
    expect(prisma.aIReview.create).toHaveBeenCalledWith({
      data: expect.objectContaining({ aiResultId: 'v1', reviewerId: 'u1', decision: 'approved', finalReport: 'Draft text', reason: null }),
    });
    expect(prisma.aIResult.update).toHaveBeenCalledWith({ where: { id: 'v1' }, data: { reviewStatus: 'approved' } });
    expect(jobs.enqueue).toHaveBeenCalledWith('forward-review', { visitId: 'v1' });
    expect(cache.bump).toHaveBeenCalledWith('k1');
    expect(audit.log).toHaveBeenCalledWith(expect.objectContaining({ action: 'visit.review', entityId: 'v1' }));
    expect(review).toEqual(expect.objectContaining({ decision: 'approved', finalReport: 'Draft text' }));
  });

  it('an edit needs the new report and a rejection needs a reason', async () => {
    const { service, prisma } = setup({ result: draft });
    await expect(service.review(ctx, 'v1', { decision: 'edited' })).rejects.toBeInstanceOf(BadRequestException);
    await expect(service.review(ctx, 'v1', { decision: 'rejected' })).rejects.toBeInstanceOf(BadRequestException);
    expect(prisma.aIReview.create).not.toHaveBeenCalled();
  });

  it('a draft is reviewed once', async () => {
    const { service, prisma } = setup({ result: { ...draft, review: { id: 'rv' } } });
    await expect(service.review(ctx, 'v1', { decision: 'rejected', reason: 'Wrong wound' })).rejects.toBeInstanceOf(ConflictException);
    expect(prisma.aIReview.create).not.toHaveBeenCalled();
  });

  it("treats another clinic's result as not found", async () => {
    const { service } = setup({ result: draft });
    await expect(service.review({ ...ctx, clinicId: 'k2' }, 'v1', { decision: 'approved' })).rejects.toBeInstanceOf(NotFoundException);
  });

  it('the model hears the decision with an opaque reviewer id', async () => {
    const review = { reviewerId: 'u1', decision: 'approved', finalReport: 'Draft text', corrections: null };
    const { service, model } = setup({ job: { modelCaseId: 'model-case-1', review } });
    await service.forwardReview('v1');
    expect(model.review).toHaveBeenCalledWith({ case_id: 'model-case-1', reviewer_id: 'u1', decision: 'approved', final_summary: 'Draft text', corrections: null });
  });
});
