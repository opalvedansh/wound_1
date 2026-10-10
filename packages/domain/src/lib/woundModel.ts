/**
 * The wound model service's contract (wound-ai/api/server.py) and the visit records built from it.
 * Shared by the NestJS API, which calls the model, and the web portal, which shows the results.
 *
 * Import through `@antigravity-project-spec-pack/domain/wound-model`, never the package index: the index also
 * re-exports a browser Supabase client.
 */

export type AnalyzeStatus = 'ok' | 'retake' | 'no_wound_found';

export interface Flag {
  level: 'urgent' | 'review';
  text: string;
}

export interface ClassResult {
  label: string;
  /** Calibrated probability of `label`; null when a clinical rule decided it, not the model. */
  prob: number | null;
  /** Top alternatives, most likely first. */
  top: [string, number][];
  /** Set when a flowchart rule decided the label (e.g. "diabetes and a foot location"). */
  rule?: string;
  /** The model's own guess, kept beside a rule's answer. */
  model?: ClassResult | null;
}

export interface Measurement {
  area_cm2: number;
  length_cm: number;
  width_cm: number;
  perimeter_cm: number;
  n_regions: number;
}

export interface AreaChange {
  previous_area_cm2?: number;
  /** Positive = smaller than last time. */
  percent_area_reduction?: number;
  days_between?: number;
}

/** One polygon per wound region; points are [x, y] scaled 0–1 by the photo's width and height. */
export type Outline = [number, number][][];

export interface IntakeQuestion {
  id: string;
  text: string;
  type: 'choice' | 'number' | 'text';
  options?: string[];
}

export type IntakeAnswers = Record<string, string | number>;

/**
 * What POST /analyze returns. Only `status` is always there: a retake carries `quality.issues`, no wound found
 * carries `flags`, and the rest appears as each trained model is added to the service.
 */
export interface AnalyzeResponse {
  status: AnalyzeStatus;
  case_id?: string;
  /** pre: as found; post: after cleaning or debridement, before the dressing (same visit). */
  phase?: PhotoPhase;
  timestamp?: string;
  /** ok: no issue at all. usable: analysed (false only for a photo with nothing to analyse, i.e. a retake).
   * warnings: what made a usable photo less reliable (low_resolution, blurry, dark, glare). */
  quality?: { ok: boolean; usable?: boolean; issues: string[]; warnings?: string[] };
  marker_found?: boolean;
  flags?: Flag[];
  wound_type?: ClassResult;
  severity?: Record<string, ClassResult>;
  tissue_pct?: Record<string, number>;
  /** The tissue mix when the model was unsure (below its confidence gate): shown, never used by the rules. */
  tissue_pct_uncertain?: Record<string, number>;
  tissue_confidence?: number;
  /** One outline per tissue class, drawn in colour over the photo. */
  tissue_outline?: Record<string, Outline> | null;
  periwound_erythema_frac?: number;
  periwound_maceration_frac?: number;
  periwound_callus_frac?: number;
  /** Classes the tissue model saw but is not yet trusted on (cross-validation): for the clinician to check. */
  tissue_untrusted?: string[];
  /** Per-channel gains from the sticker's grey patch; null when no patch was found. */
  white_balance?: [number, number, number] | null;
  measurement?: Measurement | null;
  change?: AreaChange;
  outline?: Outline | null;
  report_markdown?: string;
  follow_up_questions?: IntakeQuestion[];
  model_versions?: Record<string, string>;
}

export type PhotoPhase = 'pre' | 'post';

// ---------------------------------------------------------------- treatment report (wound-ai/wound_ai/care.py)

/** One analysed photo, reduced to what the treatment report needs (same as progress.observation in wound-ai). */
export interface Observation {
  taken_at: string;
  status: string;
  area_cm2: number | null;
  length_cm: number | null;
  width_cm: number | null;
  perimeter_cm: number | null;
  tissue_pct: Record<string, number> | null;
  periwound_erythema_frac: number | null;
  periwound_maceration_frac: number | null;
  periwound_callus_frac: number | null;
  flags: Flag[];
}

export const observation = (f: AnalyzeResponse, takenAt: string): Observation => ({
  taken_at: takenAt,
  status: f.status,
  area_cm2: f.measurement?.area_cm2 ?? null,
  length_cm: f.measurement?.length_cm ?? null,
  width_cm: f.measurement?.width_cm ?? null,
  perimeter_cm: f.measurement?.perimeter_cm ?? null,
  tissue_pct: f.tissue_pct ?? null,
  periwound_erythema_frac: f.periwound_erythema_frac ?? null,
  periwound_maceration_frac: f.periwound_maceration_frac ?? null,
  periwound_callus_frac: f.periwound_callus_frac ?? null,
  flags: f.flags ?? [],
});

/** The clinician's assessment as the care rules read it (values are the app's own options). */
export interface CareAssessment {
  exudate_level?: string | null;
  exudate_type?: string | null;
  infection_signs?: string[];
  edge_condition?: string | null;
  periwound_condition?: string | null;
  pain_level?: number | null;
}

export interface TreatmentInput {
  sequence: number;
  pre?: Observation | null;
  post?: Observation | null;
  assessment?: CareAssessment | null;
  therapy?: string[];
  dressing?: string | null;
}

export interface TreatmentReportRequest {
  /** The PRE photo's class result. */
  wound_type: ClassResult | null;
  severity?: Record<string, ClassResult>;
  intake: IntakeAnswers;
  /** Visit order; the current treatment last. */
  treatments: TreatmentInput[];
}

export type Trajectory = 'improving' | 'static' | 'deteriorating';

export interface PhotoChange {
  days: number | null;
  area_before_cm2?: number;
  area_after_cm2?: number;
  /** Positive = smaller. */
  percent_area_reduction?: number;
  cm2_per_week?: number;
  edge_advance_cm_per_week?: number;
  nonviable_before_pct?: number;
  nonviable_after_pct?: number;
}

export interface Progress {
  /** What this visit's cleaning or debridement did (PRE → POST of the same visit). */
  session: {
    area_before_cm2?: number;
    area_after_cm2?: number;
    area_note?: string | null;
    nonviable_before_pct?: number;
    nonviable_after_pct?: number;
    nonviable_removed_points?: number;
  } | null;
  /** Visit to visit, like with like: POST photos if this visit has one, else PRE photos. */
  healing: {
    phase: PhotoPhase;
    n_photos: number;
    /** The change treated as measurement noise, and where the figure comes from (a study, or the placeholder). */
    noise_band?: { smaller_pct: number; larger_pct: number; source: string };
    comparable: boolean;
    reason?: string;
    since_last?: PhotoChange;
    since_first?: PhotoChange;
    trajectory?: Trajectory | null;
    basis?: 'area' | 'tissue' | null;
    four_week?: { days: number; percent_area_reduction: number; target: number; on_track: boolean } | null;
  };
  push: { score: number; size: number; exudate: number; tissue: number; note: string } | null;
  flags: Flag[];
}

export interface CareSuggestion {
  rule_id: string;
  domain: 'T' | 'I' | 'M' | 'E' | 'R';
  /** One of the app's therapy or dressing options. */
  action: string;
  alternatives: string[];
  text: string;
  because: string[];
}

export interface Care {
  suggestions: CareSuggestion[];
  contraindications: { action: string; reason: string }[];
  checks: { rule_id: string; text: string }[];
}

export interface TreatmentReportResponse extends Care {
  progress: Progress;
  flags: Flag[];
  report_markdown: string;
  rules_version: string;
}

/** What a reviewer decided about each suggestion, stored in the review's corrections. */
export type SuggestionDecisions = Record<string, 'accepted' | 'declined'>;

/** The model refuses an analysis without these (they decide the diabetic-foot rule and the follow-ups). */
export const REQUIRED_INTAKE = ['diabetes', 'cause'] as const;

export const missingIntake = (answers: IntakeAnswers): string[] =>
  REQUIRED_INTAKE.filter((id) => answers[id] === undefined || answers[id] === '');

/**
 * Keeps only answers to the model's own choice and number questions, with values the question allows. Free text
 * (such as "current treatment") and anything else is dropped, so nothing that could identify the patient is sent.
 */
export const filterIntake = (answers: Record<string, unknown>, questions: IntakeQuestion[]): IntakeAnswers => {
  const kept: IntakeAnswers = {};
  for (const q of questions) {
    const value = answers[q.id];
    if (q.type === 'choice' && typeof value === 'string' && (q.options ?? []).includes(value)) kept[q.id] = value;
    if (q.type === 'number') {
      const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
      if (Number.isFinite(n)) kept[q.id] = n;
    }
  }
  return kept;
};

/** Below this calibrated probability a classification is reported as uncertain (same as report.py). */
export const UNCERTAIN_BELOW = 0.7;

export const isUncertain = (result: ClassResult | undefined): boolean =>
  result !== undefined && !result.rule && result.prob !== null && result.prob < UNCERTAIN_BELOW;

/** Display names for the model's wound-type labels (same as report.py). */
export const WOUND_TYPE_LABEL: Record<string, string> = {
  diabetic: 'Diabetic foot ulcer',
  pressure: 'Pressure injury',
  venous: 'Venous leg ulcer',
  surgical: 'Surgical wound',
  burn: 'Burn',
  other: 'Other wound',
  not_wound: 'No wound detected',
};

export const woundTypeName = (label: string) => WOUND_TYPE_LABEL[label] ?? label.replace(/_/g, ' ');

/** The model's option ids are snake_case ("pressure_lying_or_sitting"); show them as words. */
export const optionLabel = (option: string) => {
  const words = option.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
};

/** The most recent measured area, which the model compares the new photo against. */
export const previousMeasurement = (
  results: { area: number | null; createdAt: Date | string }[],
  now: Date = new Date(),
): { area_cm2: number; days_ago: number } | undefined => {
  const latest = results
    .filter((r): r is { area: number; createdAt: Date | string } => r.area !== null && r.area > 0)
    .map((r) => ({ area: r.area, at: new Date(r.createdAt) }))
    .sort((a, b) => b.at.getTime() - a.at.getTime())[0];
  if (!latest) return undefined;
  return { area_cm2: latest.area, days_ago: Math.max(0, Math.round((now.getTime() - latest.at.getTime()) / 86_400_000)) };
};

export type ReviewDecision = 'approved' | 'edited' | 'rejected';
export const REVIEW_DECISIONS: readonly ReviewDecision[] = ['approved', 'edited', 'rejected'];

export interface ReviewView {
  decision: ReviewDecision;
  finalReport: string | null;
  reason: string | null;
  reviewerId: string;
  createdAt: string;
}

/** One analysed photo of a wound: what the API returns and the case page lists. */
export interface VisitView {
  aiResultId: string;
  treatmentId: string;
  takenAt: string;
  /** Short-lived signed URL, or null if the photo can't be signed right now. */
  photoUrl: string | null;
  findings: AnalyzeResponse;
  intake: IntakeAnswers;
  draftReport: string | null;
  review: ReviewView | null;
}

/** What POST /cases/:id/visits answers: a saved visit, or why the photo was not analysed (nothing saved). */
export interface VisitOutcome {
  status: AnalyzeStatus;
  visit?: VisitView;
  /** Retake: what to fix in the photo. */
  issues?: string[];
  flags?: Flag[];
}

/** A draft no clinician has reviewed yet. */
export interface QueueItem {
  aiResultId: string;
  caseId: string;
  caseLocation: string;
  patient: { id: string; firstName: string; lastName: string; patientId: string };
  takenAt: string;
  urgentFlags: number;
  reviewFlags: number;
  areaCm2: number | null;
}

/** The dashboard: counts for the signed-in clinician, and the drafts waiting for review (urgent first). */
export interface Overview {
  patients: number;
  openWounds: number;
  visitsThisWeek: number;
  awaitingReview: number;
  urgentAwaitingReview: number;
  queue: QueueItem[];
}

export interface PatientSummary {
  id: string;
  firstName: string;
  lastName: string;
  patientId: string;
  sex: string;
  dateOfBirth: string;
  location: string | null;
  createdAt: string;
  cases: CaseSummary[];
}

export interface CaseSummary {
  id: string;
  location: string;
  onset: string;
  woundType: string;
  createdAt: string;
}

export interface CaseView extends CaseSummary {
  comorbidities: string[];
  patient: { id: string; firstName: string; lastName: string; patientId: string };
  /** Oldest first. */
  visits: VisitView[];
}
