/**
 * The question catalog: what clinicians are asked in the mobile app, configured by admins in the web portal.
 * Shared by the API (which enforces these rules), the admin panel (which shows the same problems before
 * saving) and the mobile app (which falls back to the defaults when it has never reached the API).
 */

/** The assessment is recorded before treatment; the care record after it. */
export type QuestionForm = 'assessment' | 'care';
export type QuestionType = 'chip_single' | 'chip_multi' | 'numeric';

export const QUESTION_FORMS: QuestionForm[] = ['assessment', 'care'];
export const QUESTION_TYPES: QuestionType[] = ['chip_single', 'chip_multi', 'numeric'];

export interface Question {
  id: string;
  form: QuestionForm;
  /** Built-in questions only: the treatment field their answer is stored in. Null for admin-added questions. */
  fieldKey: string | null;
  title: string;
  type: QuestionType;
  /** The choices for chip questions; null for numeric ones. */
  options: string[] | null;
  required: boolean;
  order: number;
  /** Hidden questions stay in the catalog but aren't asked. */
  active: boolean;
  /** Only asked from the second treatment of a case onwards. */
  followUpOnly: boolean;
}

export type QuestionAnswer = string | string[] | number;

/** An answer to an admin-added question, kept with the question as it was asked at the time. */
export interface QuestionResponse {
  questionId: string;
  title: string;
  type: QuestionType;
  answer: QuestionAnswer;
}

/** Numeric questions are answered on the same 0 to 10 scale as the pain score. */
export const NUMERIC_SCALE = { min: 0, max: 10 } as const;

export const QUESTION_LIMITS = { title: 120, option: 80, options: 40 } as const;

interface BuiltInField {
  form: QuestionForm;
  type: QuestionType;
  /** Options the app compares by value, so they can't be edited. */
  fixedOptions?: boolean;
}

/**
 * Built-in questions store their answers in named treatment fields that sync, reports and the app read.
 * They can be reworded, reordered, hidden and have their options edited, but not deleted or retyped.
 */
export const BUILT_IN_FIELDS: Record<string, BuiltInField> = {
  woundType: { form: 'assessment', type: 'chip_single' },
  woundAppearanceTrend: { form: 'assessment', type: 'chip_single', fixedOptions: true },
  exudateLevel: { form: 'assessment', type: 'chip_single' },
  exudateType: { form: 'assessment', type: 'chip_single' },
  infectionSigns: { form: 'assessment', type: 'chip_multi' },
  pain: { form: 'assessment', type: 'numeric' },
  edgeCondition: { form: 'assessment', type: 'chip_single' },
  periwoundCondition: { form: 'assessment', type: 'chip_single' },
  comorbidities: { form: 'assessment', type: 'chip_multi' },
  therapyGiven: { form: 'care', type: 'chip_multi' },
  dressingType: { form: 'care', type: 'chip_single' },
  nextVisitDate: { form: 'care', type: 'chip_single' },
};

const question = (q: Omit<Question, 'active'> & { active?: boolean }): Question => ({ active: true, ...q });

/**
 * The catalog the app shipped with. The API seeds any built-in question missing from the database from
 * this list, and the mobile app uses it until it first loads the catalog.
 */
export const DEFAULT_QUESTIONS: Question[] = [
  question({
    id: 'default_wound_type',
    form: 'assessment',
    fieldKey: 'woundType',
    title: 'WOUND TYPE',
    type: 'chip_single',
    options: ['Pressure Ulcer', 'DFU', 'VLU', 'Arterial Ulcer', 'Surgical Wound', 'Traumatic', 'Burn', 'Other'],
    required: true,
    order: 0,
    followUpOnly: false,
  }),
  question({
    id: 'default_trend',
    form: 'assessment',
    fieldKey: 'woundAppearanceTrend',
    title: 'WOUND APPEARANCE TREND',
    type: 'chip_single',
    options: ['Improving', 'Static', 'Deteriorating'],
    required: false,
    order: 1,
    followUpOnly: true,
  }),
  question({
    id: 'default_exudate_level',
    form: 'assessment',
    fieldKey: 'exudateLevel',
    title: 'EXUDATE - Volume / Level',
    type: 'chip_single',
    options: ['None', 'Scant', 'Moderate', 'Heavy'],
    required: false,
    order: 2,
    followUpOnly: false,
  }),
  question({
    id: 'default_exudate_type',
    form: 'assessment',
    fieldKey: 'exudateType',
    title: 'EXUDATE - Type',
    type: 'chip_single',
    options: ['Serous', 'Sanguineous', 'Serosanguineous', 'Purulent'],
    required: false,
    order: 3,
    followUpOnly: false,
  }),
  question({
    id: 'default_infection',
    form: 'assessment',
    fieldKey: 'infectionSigns',
    title: 'INFECTION SIGNS',
    type: 'chip_multi',
    options: ['Erythema', 'Local warmth', 'Edema', 'Purulent discharge', 'Malodor', 'Increased pain', 'Fever/systemic signs'],
    required: false,
    order: 4,
    followUpOnly: false,
  }),
  question({
    id: 'default_pain',
    form: 'assessment',
    fieldKey: 'pain',
    title: 'PAIN LEVEL (0-10)',
    type: 'numeric',
    options: null,
    required: false,
    order: 5,
    followUpOnly: false,
  }),
  question({
    id: 'default_edges',
    form: 'assessment',
    fieldKey: 'edgeCondition',
    title: 'EDGES',
    type: 'chip_single',
    options: ['Well-defined', 'Rolled', 'Undermined', 'Macerated', 'Callused'],
    required: false,
    order: 6,
    followUpOnly: false,
  }),
  question({
    id: 'default_periwound',
    form: 'assessment',
    fieldKey: 'periwoundCondition',
    title: 'PERIWOUND',
    type: 'chip_single',
    options: ['Healthy', 'Dry/Flaky', 'Macerated', 'Erythematous', 'Fragile'],
    required: false,
    order: 7,
    followUpOnly: false,
  }),
  question({
    id: 'default_comorbidities',
    form: 'assessment',
    fieldKey: 'comorbidities',
    title: 'COMORBIDITIES',
    type: 'chip_multi',
    options: ['Diabetes', 'Hypertension', 'CVD', 'CKD', 'PVD', 'Obesity', 'Smoking', 'Immunocompromised', 'None'],
    required: false,
    order: 8,
    followUpOnly: false,
  }),
  // The care questions and their options are the ones the therapy form had built in.
  question({
    id: 'default_therapy',
    form: 'care',
    fieldKey: 'therapyGiven',
    title: 'THERAPY GIVEN',
    type: 'chip_multi',
    // Offloading and pressure redistribution are the main treatments for diabetic foot ulcers and pressure
    // injuries; the care suggestions (wound-ai/wound_ai/care.py) use these exact names.
    options: [
      'Debridement',
      'Cleansing',
      'Negative Pressure (NPWT)',
      'Skin Substitute',
      'Compression',
      'Offloading',
      'Pressure redistribution',
      'None',
    ],
    required: false,
    order: 0,
    followUpOnly: false,
  }),
  question({
    id: 'default_dressing',
    form: 'care',
    fieldKey: 'dressingType',
    title: 'DRESSING TYPE',
    type: 'chip_single',
    options: ['Foam', 'Gauze', 'Alginate', 'Hydrogel', 'Hydrocolloid', 'Collagen', 'Silver', 'None'],
    required: true,
    order: 1,
    followUpOnly: false,
  }),
  question({
    id: 'default_next_visit',
    form: 'care',
    fieldKey: 'nextVisitDate',
    title: 'NEXT VISIT / ACTION DATE',
    type: 'chip_single',
    options: ['In 3 Days', '1 Week', '2 Weeks', '1 Month', 'PRN (As Needed)'],
    required: false,
    order: 2,
    followUpOnly: false,
  }),
];

/** A form's questions in the order they're asked. */
export const questionsFor = (questions: Question[], form: QuestionForm) =>
  questions.filter((q) => q.form === form).sort((a, b) => a.order - b.order);

/** What a create or update request may change. `fieldKey` and `order` are never taken from a request. */
export interface QuestionChanges {
  form?: unknown;
  title?: unknown;
  type?: unknown;
  options?: unknown;
  required?: unknown;
  active?: unknown;
  followUpOnly?: unknown;
}

export type QuestionValues = Pick<Question, 'form' | 'title' | 'type' | 'options' | 'required' | 'active' | 'followUpOnly'>;

export type QuestionCheck = { ok: true; value: QuestionValues } | { ok: false; problems: string[] };

const sameList = (a: string[] | null, b: string[] | null) =>
  (a ?? []).length === (b ?? []).length && (a ?? []).every((item, i) => item === (b ?? [])[i]);

/**
 * Checks a new question (no `current`) or changes to an existing one against the catalog rules, and
 * returns the cleaned values to store or every problem found, worded for the admin panel.
 */
export function checkQuestion(changes: QuestionChanges, current?: Question): QuestionCheck {
  const problems: string[] = [];
  const builtIn = current?.fieldKey ? BUILT_IN_FIELDS[current.fieldKey] : undefined;

  const form = current ? current.form : changes.form;
  if (current && changes.form !== undefined && changes.form !== current.form) {
    problems.push("A question can't move to another form.");
  } else if (!QUESTION_FORMS.includes(form as QuestionForm)) {
    problems.push('Choose the form this question belongs to.');
  }

  const rawTitle = changes.title !== undefined ? changes.title : current?.title;
  const title = typeof rawTitle === 'string' ? rawTitle.trim().replace(/\s+/g, ' ') : '';
  if (!title) problems.push('Enter the question.');
  else if (title.length > QUESTION_LIMITS.title) problems.push(`Keep the question under ${QUESTION_LIMITS.title} characters.`);

  const type = changes.type !== undefined ? changes.type : current?.type;
  if (!QUESTION_TYPES.includes(type as QuestionType)) problems.push('Choose how the question is answered.');
  else if (builtIn && type !== builtIn.type) problems.push("Built-in questions keep their answer type.");

  let options: string[] | null = null;
  if (type === 'chip_single' || type === 'chip_multi') {
    const raw = changes.options !== undefined ? changes.options : current?.options;
    if (raw !== null && raw !== undefined && !Array.isArray(raw)) {
      problems.push('Options must be a list.');
    } else {
      options = ((raw ?? []) as unknown[])
        .map((option) => (typeof option === 'string' ? option.trim().replace(/\s+/g, ' ') : ''))
        .filter(Boolean);
      const seen = new Set<string>();
      const duplicate = options.some((option) => {
        const key = option.toLowerCase();
        if (seen.has(key)) return true;
        seen.add(key);
        return false;
      });
      if (options.length < 2) problems.push('Add at least two options.');
      if (options.length > QUESTION_LIMITS.options) problems.push(`Use ${QUESTION_LIMITS.options} options or fewer.`);
      if (options.some((option) => option.length > QUESTION_LIMITS.option)) {
        problems.push(`Keep each option under ${QUESTION_LIMITS.option} characters.`);
      }
      if (duplicate) problems.push('Each option must be different.');
      if (builtIn?.fixedOptions && current && !sameList(options, current.options)) {
        problems.push("This question's options can't change: the app compares these exact values.");
      }
    }
  }

  const flag = (name: 'required' | 'active' | 'followUpOnly', fallback: boolean) => {
    const value = changes[name] !== undefined ? changes[name] : current ? current[name] : fallback;
    if (typeof value !== 'boolean') {
      problems.push(`"${name}" must be true or false.`);
      return fallback;
    }
    return value;
  };
  const required = flag('required', false);
  const active = flag('active', true);
  const followUpOnly = flag('followUpOnly', false);

  if (problems.length > 0) return { ok: false, problems };
  return {
    ok: true,
    value: { form: form as QuestionForm, title, type: type as QuestionType, options, required, active, followUpOnly },
  };
}
