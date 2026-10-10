import { BadGatewayException, HttpException, Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import type {
  AnalyzeResponse,
  IntakeAnswers,
  IntakeQuestion,
  PhotoPhase,
  ReviewDecision,
  TreatmentReportRequest,
  TreatmentReportResponse,
} from '@antigravity-project-spec-pack/domain/wound-model';

const ANALYZE_TIMEOUT_MS = 60_000; // a hosted model that has gone to sleep can take most of this to start
const QUESTIONS_TIMEOUT_MS = 15_000;
const CORE_QUESTIONS_TTL_MS = 10 * 60_000;

export interface ModelReview {
  case_id: string;
  reviewer_id: string;
  decision: ReviewDecision;
  final_summary: string | null;
  corrections: Record<string, unknown> | null;
}

/**
 * Calls the wound model service (wound-ai/api/server.py). Only this server holds WOUND_API_KEY; browsers never
 * reach the model. Logs carry paths and statuses only, never photos or answers.
 */
@Injectable()
export class ModelClient {
  private readonly logger = new Logger(ModelClient.name);
  private core?: { questions: IntakeQuestion[]; at: number };

  async questions(): Promise<IntakeQuestion[]> {
    if (this.core && Date.now() - this.core.at < CORE_QUESTIONS_TTL_MS) return this.core.questions;
    const questions = (await (await this.call('/intake/questions', () => ({ method: 'GET' }), QUESTIONS_TIMEOUT_MS)).json()) as IntakeQuestion[];
    this.core = { questions, at: Date.now() };
    return questions;
  }

  async followUps(answers: IntakeAnswers, predictedType?: string): Promise<IntakeQuestion[]> {
    const res = await this.call(
      '/intake/follow-ups',
      () => ({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers, predicted_type: predictedType ?? null }),
      }),
      QUESTIONS_TIMEOUT_MS,
    );
    return (await res.json()) as IntakeQuestion[];
  }

  async analyze(
    photo: { buffer: Buffer; mimetype: string },
    intake: IntakeAnswers,
    previous?: { area_cm2: number; days_ago: number },
    phase: PhotoPhase = 'pre',
  ): Promise<AnalyzeResponse> {
    const res = await this.call(
      '/analyze',
      () => {
        // Built per attempt: a request body can only be sent once.
        const form = new FormData();
        form.append('image', new Blob([new Uint8Array(photo.buffer)], { type: photo.mimetype }), 'photo');
        form.append('intake', JSON.stringify(intake));
        if (previous) form.append('previous', JSON.stringify(previous));
        form.append('phase', phase);
        return { method: 'POST', body: form };
      },
      ANALYZE_TIMEOUT_MS,
    );
    return (await res.json()) as AnalyzeResponse;
  }

  /** Healing, care suggestions and the draft for a treatment, from findings already stored (no photos sent). */
  async treatmentReport(req: TreatmentReportRequest): Promise<TreatmentReportResponse> {
    const res = await this.call(
      '/treatment-report',
      () => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) }),
      QUESTIONS_TIMEOUT_MS,
    );
    return (await res.json()) as TreatmentReportResponse;
  }

  /** Tells the model service about the clinician's decision (kept only in a consented study). Never fails the caller. */
  async review(review: ModelReview): Promise<void> {
    try {
      await this.call(
        '/review',
        () => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(review) }),
        QUESTIONS_TIMEOUT_MS,
      );
    } catch {
      // Already logged; the clinician's review is saved in our database either way.
    }
  }

  private async call(path: string, request: () => RequestInit, timeoutMs: number): Promise<Response> {
    const base = process.env['WOUND_API_URL']?.replace(/\/+$/, '');
    const key = process.env['WOUND_API_KEY'];
    if (!base || !key) {
      this.logger.error('WOUND_API_URL or WOUND_API_KEY is not set, so the wound model cannot be called.');
      throw new ServiceUnavailableException('The wound model is not set up on this server.');
    }

    for (let attempt = 1; ; attempt++) {
      try {
        const init = request();
        const res = await fetch(`${base}${path}`, {
          ...init,
          headers: { ...(init.headers as Record<string, string> | undefined), 'X-API-Key': key },
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (res.ok) return res;
        this.logger.error(`Wound model ${path} answered ${res.status}${res.status === 401 || res.status === 503 ? ' (check WOUND_API_KEY on both servers)' : ''}.`);
        throw res.status === 413
          ? new BadGatewayException('The photo is too large for the wound model.')
          : new BadGatewayException("The wound model couldn't process this request.");
      } catch (error) {
        if (error instanceof HttpException) throw error;
        // A timeout or refused connection: a model that was asleep is often ready on the second try.
        if (attempt === 1) continue;
        this.logger.error(`Wound model ${path} unreachable: ${error instanceof Error ? error.name : 'error'}.`);
        throw new ServiceUnavailableException("The wound model isn't responding. Try again in a minute.");
      }
    }
  }
}
