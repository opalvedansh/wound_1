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
  /** Where the scale came from: the calibration sticker, or the distance the phone measured when it took the photo. */
  method?: 'sticker' | 'phone_distance';
}

export interface AreaChange {
  previous_area_cm2?: number;
  /** Positive = smaller than last time. */
  percent_area_reduction?: number;
  days_between?: number;
}

/**
 * Redness of the skin around the wound, from the photo's colour: how much redder (CIELAB a*) the skin beside the
 * wound is than skin further out. A hint only: it shows less on darker skin, and no rule uses it.
 */
export interface Redness {
  delta_a: number;
  level: 'none' | 'mild' | 'marked';
  /** The photo's colour cast was corrected from the sticker's grey patch. */
  white_balanced: boolean;
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

/** The distance the phone measured when it took the photo, which gives a photo without a sticker its scale. */
export interface PhoneReading {
  distance_mm: number;
  /** The camera's field of view across the photo's width. */
  hfov_deg: number;
  /** Which way the surface around the wound faces (photo axes: x right, y down, z into the scene). With it the
   * model corrects for the phone's tilt. */
  normal?: [number, number, number];
  /** How far that surface is from flat: large on curved skin, where the size is under-estimated. */
  surface_rms_mm?: number;
  source?: string;
}

/** What POST /check returns: the capture checks alone, before any model runs. */
export interface PhotoCheck {
  quality: NonNullable<AnalyzeResponse['quality']>;
  marker_found: boolean;
  phone_reading: PhoneReading | null;
}

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
  phone_reading?: PhoneReading | null;
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
  periwound_redness?: Redness | null;
  /** Classes the tissue model saw but is not yet trusted on (cross-validation): for the clinician to check. */
  tissue_untrusted?: string[];
  /** Per-channel gains from the sticker's grey patch; null when no patch was found. */
  white_balance?: [number, number, number] | null;
  measurement?: Measurement | null;
  /** A photo with both the sticker and a phone reading: how far the phone's area is from the sticker's. */
  measurement_check?: { sticker_area_cm2: number; phone_area_cm2: number; phone_vs_sticker_pct: number };
  change?: AreaChange;
  outline?: Outline | null;
  report_markdown?: string;
  follow_up_questions?: IntakeQuestion[];
  model_versions?: Record<string, string>;
}

export type PhotoPhase = 'pre' | 'post';

/** A photo's findings as the phone gets them: everything but the outlines, the draft and the answers. */
export type PhotoFindings = Omit<AnalyzeResponse, 'outline' | 'tissue_outline' | 'report_markdown' | 'follow_up_questions'>;

export const photoFindings = (f: AnalyzeResponse): PhotoFindings => {
  const { outline: _o, tissue_outline: _t, report_markdown: _r, follow_up_questions: _q, ...rest } = f;
  return rest;
};

/** A post photo taken within this many hours of the pre photo is the same visit (same as progress.py). */
export const SAME_VISIT_HOURS = 12;

/** Were a treatment's two photos taken in one visit? Photos without dates count as one visit. */
export const sameVisit = (preTakenAt: string | null | undefined, postTakenAt: string | null | undefined): boolean => {
  const hours = (new Date(postTakenAt ?? '').getTime() - new Date(preTakenAt ?? '').getTime()) / 3_600_000;
  return Number.isNaN(hours) || Math.abs(hours) < SAME_VISIT_HOURS;
};

/** What to call a treatment's post photo: taken after cleaning in the same visit, or days later. */
export const postPhotoLabel = (preTakenAt: string | null | undefined, postTakenAt: string | null | undefined): string => {
  if (sameVisit(preTakenAt, postTakenAt)) return 'After cleaning';
  const days = Math.round((new Date(postTakenAt ?? '').getTime() - new Date(preTakenAt ?? '').getTime()) / 86_400_000);
  return `After treatment, ${days} ${days === 1 ? 'day' : 'days'} later`;
};

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
  periwound_redness: Redness | null;
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
  periwound_redness: f.periwound_redness ?? null,
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
  /** Probed by the clinician: a photo cannot show depth. */
  depth_cm?: number | null;
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
  /** What this visit's cleaning or debridement did (PRE → POST of the same visit). Null when the POST photo is
   * from days later. */
  session: {
    area_before_cm2?: number;
    area_after_cm2?: number;
    area_note?: string | null;
    nonviable_before_pct?: number;
    nonviable_after_pct?: number;
    nonviable_removed_points?: number;
  } | null;
  /** What the treatment had done by a POST photo taken days after the PRE photo. */
  response?: (PhotoChange & { trajectory: Trajectory | null; basis: 'area' | 'tissue' | null }) | null;
  /** The depth the clinician probed at this visit, beside the last one recorded. */
  depth?: { depth_cm: number; previous_cm?: number } | null;
  /** Visit to visit, like with like: after-cleaning photos ("post"), else the wound as found ("pre": every PRE
   * photo and any POST photo taken days after its visit). */
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

// ---------------------------------------------------------------- the wound analysis in words (portal and phone)

export interface Fact {
  label: string;
  value: string;
}

const pct = (p: number) => `${Math.round(p * 100)}%`;
const words = (id: string) => id.replace(/_/g, ' ');

export const SEVERITY_NAME: Record<string, string> = {
  pu_stage: 'Pressure injury stage',
  burn_depth: 'Burn depth',
  dfu_wagner: 'Wagner grade',
  dfu_infection: 'Infection/ischaemia',
};

/** A classification as the report writes it: a confident label, or uncertain with the top estimates. */
export function classText(result: ClassResult | undefined | null, name: (label: string) => string): string {
  if (!result) return 'Not assessed yet (model not installed)';
  if (result.rule) {
    return `${name(result.label)} (by rule: ${result.rule}${result.model ? `; model estimate: ${classText(result.model, name)}` : ''})`;
  }
  if (result.prob === null) return name(result.label);
  if (isUncertain(result)) {
    return `Uncertain. Top estimates: ${result.top.map(([label, p]) => `${name(label)} ${pct(p)}`).join(', ')}`;
  }
  return `${name(result.label)} (model confidence ${pct(result.prob)})`;
}

const REDNESS_TEXT: Record<Redness['level'], string> = { none: 'None seen', mild: 'Mild', marked: 'Marked' };

/**
 * What the model found in one photo, as label and value rows: type, grade, size, depth, tissue, redness. Depth is
 * the clinician's (a photo cannot show it); pass it when the assessment recorded one.
 */
export function woundFacts(f: PhotoFindings, depthCm?: number | null): Fact[] {
  const m = f.measurement;
  const tissue = (t: Record<string, number>) => Object.entries(t).map(([k, v]) => `${words(k)} ${v}%`).join(', ');
  const rows: Fact[] = [
    { label: 'Wound type', value: classText(f.wound_type, woundTypeName) },
    ...Object.entries(f.severity ?? {}).map(([head, result]) => ({ label: SEVERITY_NAME[head] ?? words(head), value: classText(result, words) })),
    {
      label: 'Size',
      value: m
        ? `${m.area_cm2} cm² · ${m.length_cm} × ${m.width_cm} cm · perimeter ${m.perimeter_cm} cm`
        : f.marker_found === false
          ? 'Not measured: calibration sticker not found in the photo'
          : 'Not measured: no wound outline yet (outline model not installed)',
    },
    {
      label: 'Depth',
      value: depthCm !== null && depthCm !== undefined ? `${depthCm} cm (probed by the clinician)` : 'Not recorded. A photo cannot show depth: probe it and enter it in the assessment.',
    },
  ];
  if (f.tissue_pct) rows.push({ label: 'Tissue', value: tissue(f.tissue_pct) });
  else if (f.tissue_pct_uncertain) {
    rows.push({ label: 'Tissue', value: `Uncertain (model confidence ${pct(f.tissue_confidence ?? 0)}): ${tissue(f.tissue_pct_uncertain)}. Assess on examination.` });
  }
  if (f.tissue_untrusted?.length) {
    rows.push({ label: 'Possibly also', value: `${f.tissue_untrusted.map(words).join(', ')} (the tissue model is not yet reliable for these: check on examination)` });
  }
  if (f.periwound_erythema_frac !== undefined && f.periwound_erythema_frac !== null) {
    rows.push({ label: 'Redness around the wound', value: `Red skin on ${pct(f.periwound_erythema_frac)} of the skin beside the wound` });
  } else if (f.periwound_redness) {
    rows.push({
      label: 'Redness around the wound',
      value: `${REDNESS_TEXT[f.periwound_redness.level]} in the photo's colour. A hint only: redness shows less on darker skin, so check on examination.`,
    });
  }
  return rows;
}

/** A change between two photos in words, or null when neither size nor tissue could be compared. */
export const changeText = (c: PhotoChange | null | undefined): string | null => {
  if (!c) return null;
  const parts: string[] = [];
  if (c.percent_area_reduction !== undefined) {
    parts.push(`${c.area_before_cm2} → ${c.area_after_cm2} cm² (${Math.abs(c.percent_area_reduction)}% ${c.percent_area_reduction >= 0 ? 'smaller' : 'larger'})`);
    if (c.cm2_per_week !== undefined) parts.push(`${c.cm2_per_week} cm²/week`);
  }
  if (c.nonviable_before_pct !== undefined) parts.push(`non-viable tissue ${c.nonviable_before_pct}% → ${c.nonviable_after_pct}%`);
  if (!parts.length) return null;
  return `${parts.join(' · ')}${c.days ? ` over ${c.days} days` : ''}`;
};

export const TRAJECTORY_NAME: Record<Trajectory, string> = { improving: 'Improving', static: 'Static', deteriorating: 'Deteriorating' };

/** baseline: the first analysed photo, which later ones are compared with. not_compared: nothing comparable. */
export type HealingState = Trajectory | 'baseline' | 'not_compared';

export interface HealingVerdict {
  state: HealingState;
  title: string;
  /** How it was judged, or why it could not be. */
  detail: string;
  rows: Fact[];
}

/** Is the wound healing or getting worse, with the figures behind the answer. Null until the report has run. */
export function healingVerdict(progress: Progress | null | undefined): HealingVerdict | null {
  if (!progress) return null;
  const h = progress.healing;
  const s = progress.session;
  const r = progress.response;
  const session = s
    ? [
        s.area_before_cm2 !== undefined ? `${s.area_before_cm2} → ${s.area_after_cm2} cm²${s.area_note ? ` (${s.area_note})` : ''}` : null,
        s.nonviable_before_pct !== undefined ? `non-viable tissue ${s.nonviable_before_pct}% → ${s.nonviable_after_pct}%` : null,
      ]
        .filter(Boolean)
        .join(' · ') || null
    : null;
  const responseText = r ? [r.trajectory ? TRAJECTORY_NAME[r.trajectory] : null, changeText(r)].filter(Boolean).join(': ') || null : null;
  const candidates: [string, string | null][] = [
    ['Before → after this treatment', responseText],
    // With a later post photo, "since last" is that same before → after pair.
    ['Since last visit', r ? null : changeText(h.since_last)],
    ['Since first visit', changeText(h.since_first)],
    [
      '4-week check',
      h.four_week
        ? `${h.four_week.percent_area_reduction}% smaller at day ${h.four_week.days} (target ${h.four_week.target}%): ${h.four_week.on_track ? 'on track' : 'not on track'}`
        : null,
    ],
    ["This visit's cleaning", session],
    ['Depth', progress.depth ? `${progress.depth.depth_cm} cm${progress.depth.previous_cm !== undefined ? ` (was ${progress.depth.previous_cm} cm)` : ''}` : null],
    ['PUSH score', progress.push ? `${progress.push.score}/17 (${progress.push.note})` : null],
  ];
  const rows = candidates.filter((c): c is [string, string] => c[1] !== null).map(([label, value]) => ({ label, value }));
  if (h.trajectory) {
    const noise = h.noise_band ? ` (${h.noise_band.smaller_pct}% smaller to ${h.noise_band.larger_pct}% larger; ${h.noise_band.source})` : '';
    return {
      state: h.trajectory,
      title: TRAJECTORY_NAME[h.trajectory],
      detail: `By ${h.basis === 'area' ? 'wound area' : 'tissue mix'}, comparing ${h.phase === 'post' ? 'after-cleaning photos' : 'photos of the wound as found'}. Changes within measurement noise${noise} count as static.`,
      rows,
    };
  }
  if (h.n_photos <= 1) {
    return { state: 'baseline', title: 'Baseline', detail: 'The first analysed photo of this wound. Later photos are compared with it.', rows };
  }
  return { state: 'not_compared', title: 'Not compared', detail: h.reason ?? 'Not enough photos to compare.', rows };
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
