# Repeat-photo study: how much does a wound's measured size change when nothing has changed?

**For:** the clinic staff who take the photos, and whoever runs the analysis.
**Time:** about 2 extra minutes per wound, over a few clinic days.

## Why

The app reports a wound as improving or deteriorating only when the change in size is bigger than the measurement noise. Some noise always comes from re-photographing the same wound: a slightly different angle or distance, different light, where the sticker sits, how the outline is traced. Until this study is done, the app uses a guess of ±15%. This study replaces that guess with the real figure for your staff, your phones and your patients.

## What to do

1. **Wounds:** at least **30 wounds** (40 to 50 is better), from different patients, with a mix of:
   - sizes (small, medium, large);
   - wound types;
   - body sites, including **heels and toes**, because curved sites are usually noisier;
   - skin tones.

   Count each wound once.
2. **Photographers:** two staff members (A and B), each working **independently**. Each one places the calibration sticker themselves, frames the photo themselves and takes it with their own phone, following the app's capture guide. Don't watch each other or copy the framing.
3. **Timing:** both photos are taken in the **same visit**, one straight after the other, **after cleaning and before the dressing**. Nothing may be done to the wound between the two photos.
4. **What not to do:**
   - don't retake a photo to make it match the other one;
   - don't throw away bad photos (the analysis records why a photo couldn't be measured, and that is part of the result).
5. **Naming:** `<wound>_<photographer>.jpg`, for example `W017_A.jpg` and `W017_B.jpg`. Use a study code for the wound, never the patient's name or ID.
6. **Body sites (optional but useful):** a sheet `sites.csv` with columns `wound,body_location`, for example `W017,heel`. Use the app's site names: foot_plantar, toe, heel, ankle, lower_leg, sacrum_buttock and so on.

**Consent and privacy:** these are patient photos. Follow your clinic's consent process for using photos to evaluate the system. Keep the files on clinic storage, not on personal phones once the photos have been copied off.

## Running the analysis

```bash
cd wound-ai
python scripts/repeatability.py --photos /path/to/study_photos --sites /path/to/sites.csv
```

It measures every photo exactly as the app would, then reports:
- how many photos were excluded, and why (no sticker, blurry, no wound found);
- the noise band: how much smaller or larger a wound must be before it counts as changed;
- the band per body site.

When there are at least 30 wounds with two usable photos, run it again with `--write`. This saves `wound_ai/measurement_noise.json`, and from then on the app's healing reports use the measured band and say so ("repeat-photo study <date> (<n> wounds)").

## Then

1. **Commit** `wound_ai/measurement_noise.json`. It ships with the code.
2. **Record the figure** in item H1 of [clinical_signoff.md](clinical_signoff.md) for the clinician's sign-off.
3. **Check the curved sites.** If the band for heels or toes is much wider than the overall band, tell the clinician. Size changes at those sites need more caution.
4. **Repeat the study** when the outline model is retrained, or when the clinic changes phones. The band belongs to the system that was tested.
