import {
  filterIntake,
  healingVerdict,
  isUncertain,
  missingIntake,
  optionLabel,
  photoFindings,
  postPhotoLabel,
  previousMeasurement,
  sameVisit,
  woundFacts,
  woundTypeName,
  type AnalyzeResponse,
  type IntakeQuestion,
  type Progress,
} from './woundModel';

describe('missingIntake', () => {
  it('names the answers the model refuses to work without', () => {
    expect(missingIntake({})).toEqual(['diabetes', 'cause']);
    expect(missingIntake({ diabetes: 'no', cause: '' })).toEqual(['cause']);
    expect(missingIntake({ diabetes: 'no', cause: 'burn' })).toEqual([]);
  });
});

describe('filterIntake', () => {
  const questions: IntakeQuestion[] = [
    { id: 'diabetes', text: 'Diabetes?', type: 'choice', options: ['yes', 'no', 'not_sure'] },
    { id: 'pain', text: 'Pain 0-10?', type: 'number' },
    { id: 'current_treatment', text: 'Current dressing (free text)', type: 'text' },
  ];

  it('keeps valid answers to the model questions only', () => {
    expect(
      filterIntake(
        { diabetes: 'yes', pain: '6', current_treatment: "Mrs Rao's foam dressing", patient_name: 'Asha Rao' },
        questions,
      ),
    ).toEqual({ diabetes: 'yes', pain: 6 });
  });

  it('drops options the question does not offer and numbers that are not numbers', () => {
    expect(filterIntake({ diabetes: 'maybe', pain: 'lots' }, questions)).toEqual({});
    expect(filterIntake({ pain: 0 }, questions)).toEqual({ pain: 0 });
  });
});

describe('previousMeasurement', () => {
  const now = new Date('2026-10-08T12:00:00Z');

  it('uses the latest measured area and how many days ago it was', () => {
    expect(
      previousMeasurement(
        [
          { area: 6.2, createdAt: '2026-09-10T12:00:00Z' },
          { area: 4.8, createdAt: '2026-09-24T12:00:00Z' },
          { area: null, createdAt: '2026-10-01T12:00:00Z' },
        ],
        now,
      ),
    ).toEqual({ area_cm2: 4.8, days_ago: 14 });
  });

  it('has nothing to compare against before a size was measured', () => {
    expect(previousMeasurement([], now)).toBeUndefined();
    expect(previousMeasurement([{ area: null, createdAt: now }, { area: 0, createdAt: now }], now)).toBeUndefined();
  });
});

describe('display helpers', () => {
  it('reports a classification below 70% as uncertain', () => {
    expect(isUncertain({ label: 'diabetic', prob: 0.69, top: [] })).toBe(true);
    expect(isUncertain({ label: 'diabetic', prob: 0.7, top: [] })).toBe(false);
    expect(isUncertain(undefined)).toBe(false);
    expect(isUncertain({ label: 'diabetic', prob: null, top: [], rule: 'diabetes and a foot location' })).toBe(false);
  });

  it('turns model ids into words', () => {
    expect(woundTypeName('pressure')).toBe('Pressure injury');
    expect(woundTypeName('arterial_ulcer')).toBe('arterial ulcer');
    expect(optionLabel('pressure_lying_or_sitting')).toBe('Pressure lying or sitting');
  });
});

describe('woundFacts', () => {
  const findings: AnalyzeResponse = {
    status: 'ok',
    wound_type: { label: 'venous', prob: 0.91, top: [['venous', 0.91]] },
    severity: { dfu_wagner: { label: 'grade_2', prob: 0.8, top: [['grade_2', 0.8]] } },
    measurement: { area_cm2: 4.2, length_cm: 3, width_cm: 2, perimeter_cm: 8, n_regions: 1 },
    tissue_pct: { granulation: 70, slough: 30 },
    periwound_redness: { delta_a: 9, level: 'marked', white_balanced: true },
    outline: [[[0, 0], [1, 0], [1, 1]]],
  };
  const value = (label: string, depth?: number | null) => woundFacts(findings, depth).find((r) => r.label === label)?.value;

  it('states type, grade, size, tissue and redness from the photo', () => {
    expect(value('Wound type')).toBe('Venous leg ulcer (model confidence 91%)');
    expect(value('Wagner grade')).toBe('grade 2 (model confidence 80%)');
    expect(value('Size')).toContain('4.2 cm²');
    expect(value('Tissue')).toBe('granulation 70%, slough 30%');
    expect(value('Redness around the wound')).toMatch(/^Marked .*darker skin/);
  });

  it("takes depth from the clinician, never the photo", () => {
    expect(value('Depth', 0.8)).toBe('0.8 cm (probed by the clinician)');
    expect(value('Depth')).toContain('A photo cannot show depth');
  });

  it('leaves the outlines out of what the phone gets', () => {
    expect(photoFindings(findings)).not.toHaveProperty('outline');
    expect(photoFindings(findings).measurement?.area_cm2).toBe(4.2);
  });
});

describe('healingVerdict', () => {
  const noise = { smaller_pct: 15, larger_pct: 15, source: 'placeholder' };
  const base = { session: null, push: null, flags: [] };

  it('calls the first photo the baseline', () => {
    const p: Progress = { ...base, healing: { phase: 'pre', n_photos: 1, comparable: false, reason: 'first analysed photo of this wound' } };
    expect(healingVerdict(p)?.state).toBe('baseline');
    expect(healingVerdict(null)).toBeNull();
  });

  it('says whether the wound is healing, with the figures', () => {
    const change = { days: 7, area_before_cm2: 10, area_after_cm2: 6, percent_area_reduction: 40 };
    const p: Progress = {
      ...base,
      depth: { depth_cm: 0.8, previous_cm: 1.2 },
      healing: { phase: 'pre', n_photos: 2, comparable: true, noise_band: noise, since_last: change, since_first: change, trajectory: 'improving', basis: 'area' },
    };
    const v = healingVerdict(p);
    expect(v?.title).toBe('Improving');
    expect(v?.rows.find((r) => r.label === 'Since last visit')?.value).toBe('10 → 6 cm² (40% smaller) over 7 days');
    expect(v?.rows.find((r) => r.label === 'Depth')?.value).toBe('0.8 cm (was 1.2 cm)');
  });

  it('reports a post photo from days later as the response to the treatment', () => {
    const change = { days: 9, area_before_cm2: 9, area_after_cm2: 12, percent_area_reduction: -33.3 };
    const p: Progress = {
      ...base,
      response: { ...change, trajectory: 'deteriorating', basis: 'area' },
      healing: { phase: 'pre', n_photos: 2, comparable: true, noise_band: noise, since_last: change, since_first: change, trajectory: 'deteriorating', basis: 'area' },
    };
    const v = healingVerdict(p);
    expect(v?.state).toBe('deteriorating');
    expect(v?.rows[0]).toEqual({ label: 'Before → after this treatment', value: 'Deteriorating: 9 → 12 cm² (33.3% larger) over 9 days' });
    expect(v?.rows.some((r) => r.label === 'Since last visit')).toBe(false);
  });
});

describe('sameVisit', () => {
  it('tells a post photo taken after cleaning from one taken days later', () => {
    expect(sameVisit('2026-01-01T10:00:00Z', '2026-01-01T10:20:00Z')).toBe(true);
    expect(sameVisit('2026-01-01T10:00:00Z', '2026-01-10T10:00:00Z')).toBe(false);
    expect(sameVisit('2026-01-01T10:00:00Z', null)).toBe(true);
    expect(postPhotoLabel('2026-01-01T10:00:00Z', '2026-01-01T10:20:00Z')).toBe('After cleaning');
    expect(postPhotoLabel('2026-01-01T10:00:00Z', '2026-01-10T10:00:00Z')).toBe('After treatment, 9 days later');
  });
});
