import { intakeFromApp, withMeasuredLength } from './sync.service';

describe('intakeFromApp', () => {
  it('derives the body site, diabetes and cause from what the app records', () => {
    expect(intakeFromApp('Left heel', { woundType: 'Pressure injury', comorbidities: ['Type 2 diabetes'], pain: 4 }, [])).toEqual({
      body_location: 'heel',
      diabetes: 'yes',
      cause: 'pressure_lying_or_sitting',
      pain: 4,
    });
    expect(intakeFromApp('Sacrum', null, ['Diabetes mellitus'])).toEqual({ body_location: 'sacrum_buttock', diabetes: 'yes', cause: 'unknown' });
  });

  it("says it doesn't know rather than guessing", () => {
    expect(intakeFromApp('Somewhere', { woundType: '', comorbidities: [] }, [])).toEqual({ body_location: 'other', diabetes: 'not_sure', cause: 'unknown' });
  });
});

describe('withMeasuredLength', () => {
  const intake = { diabetes: 'no', cause: 'unknown' };

  it("adds the wound's ruler length as the photo's form field sends it", () => {
    expect(withMeasuredLength(intake, '3.2')).toEqual({ ...intake, measured_length_cm: 3.2 });
    expect(withMeasuredLength(intake, 4)).toEqual({ ...intake, measured_length_cm: 4 });
  });

  it('leaves the answers alone when no usable length came with the photo', () => {
    for (const raw of [undefined, '', '  ', 'long', '-2', 0, null, ['3']]) expect(withMeasuredLength(intake, raw)).toBe(intake);
  });
});
