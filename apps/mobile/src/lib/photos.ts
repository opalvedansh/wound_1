import type { Treatment } from '@antigravity-project-spec-pack/domain';
import {
  TRAJECTORY_NAME,
  healingVerdict,
  postPhotoLabel,
  woundFacts,
  woundTypeName,
  type Fact,
  type Flag,
  type HealingVerdict,
} from '@antigravity-project-spec-pack/domain/wound-model';

/** The photo to show: the copy on this device, else the server's (taken on another device or before a reinstall). */
export const prePhoto = (t: Treatment | undefined) => t?.preImageUri ?? t?.remote?.preUrl ?? undefined;
export const postPhoto = (t: Treatment | undefined) => t?.postImageUri ?? t?.remote?.postUrl ?? undefined;

const POST_STATUS = {
  processing: 'Analysing…',
  ok: 'Analysed',
  retake: 'Kept, but the model could not use it',
  no_wound_found: 'Kept, no wound found in it',
  failed: 'Analysis failed',
} as const;

/** What to call the post photo: after cleaning in the same visit, or days later. Undefined until there is one. */
export const postLabel = (t: Treatment): string | undefined => {
  if (!postPhoto(t)) return undefined;
  const ai = t.remote?.ai;
  return postPhotoLabel(ai?.takenAt ?? t.imageMetadata?.takenAt?.pre, ai?.postTakenAt ?? t.imageMetadata?.takenAt?.post);
};

/** The wound model's analysis of a treatment, in words, for the treatment screen. */
export interface WoundAnalysis {
  /** Why there is nothing to show yet. */
  notice?: string;
  flags: Flag[];
  /** What the model found in the pre-treatment photo: type, grade, size, depth, tissue, redness. */
  before: Fact[];
  /** The post-treatment photo's size, tissue and redness, once it is analysed. */
  after: { label: string; facts: Fact[] } | null;
  /** Healing or getting worse, with the figures. */
  healing: HealingVerdict | null;
  /** Review state, and the care suggestions a clinician approved. */
  status: Fact[];
}

const notice = (text: string): WoundAnalysis => ({ notice: text, flags: [], before: [], after: null, healing: null, status: [] });

/** Treatments synced before the full healing figures were sent carry only the verdict and one figure. */
const olderVerdict = (ai: NonNullable<NonNullable<Treatment['remote']>['ai']>): HealingVerdict | null => {
  if (!ai.trajectory) return null;
  const since = ai.areaReductionSinceFirstPct;
  return {
    state: ai.trajectory,
    title: TRAJECTORY_NAME[ai.trajectory],
    detail: 'Compared with the last visit.',
    rows: since !== null && since !== undefined ? [{ label: 'Since first visit', value: `${Math.abs(since)}% ${since >= 0 ? 'smaller' : 'larger'}` }] : [],
  };
};

/** Only what can differ from the pre photo: the type and grade are the wound's, not the photo's. */
const AFTER_LABELS = ['Size', 'Tissue', 'Possibly also', 'Redness around the wound'];

export function woundAnalysis(t: Treatment): WoundAnalysis | null {
  const ai = t.remote?.ai;
  if (!ai) {
    if (t.preImageUri && !t.remote?.preStored) return notice(t.phase === 'PRE' ? 'Starts after the assessment is saved.' : 'Waiting to upload the photo.');
    return null;
  }
  if (ai.status === 'processing') return notice('Analysing the photo…');
  if (ai.status === 'failed') return notice('The analysis failed. A doctor can retry it from the portal.');
  if (ai.status === 'retake') return notice('The photo could not be used (blank, too small, or too dark or bright). Take it again.');
  if (ai.status === 'no_wound_found') return notice('No wound found in the photo.');

  const depth = t.phase === 'PRE' ? undefined : t.assessment?.depthCm;
  // Treatments synced before the findings were sent only carry the area and the type.
  const before: Fact[] = ai.findings
    ? woundFacts(ai.findings, depth)
    : [
        { label: 'Wound type', value: ai.woundType ? woundTypeName(ai.woundType) : 'Not assessed' },
        { label: 'Size', value: ai.areaCm2 !== null ? `${ai.areaCm2} cm²` : 'Not measured (no sticker found)' },
      ];
  const flags = [...(ai.findings?.flags ?? []), ...(ai.progress?.flags ?? []).filter((p) => !ai.findings?.flags?.some((f) => f.text === p.text))].sort(
    (a, b) => (a.level === b.level ? 0 : a.level === 'urgent' ? -1 : 1),
  );
  const after =
    ai.postStatus === 'ok' && ai.postFindings
      ? { label: postLabel(t) ?? 'After treatment', facts: woundFacts(ai.postFindings).filter((f) => AFTER_LABELS.includes(f.label)) }
      : null;
  return {
    flags: flags.length || !ai.urgent ? flags : [{ level: 'urgent', text: 'See the draft in the portal.' }],
    before,
    after,
    healing: healingVerdict(ai.progress) ?? olderVerdict(ai),
    status: [
      ...(ai.postStatus && ai.postStatus !== 'ok' ? [{ label: 'Post-treatment photo', value: POST_STATUS[ai.postStatus] }] : []),
      { label: 'Clinician review', value: ai.review ? ai.review[0].toUpperCase() + ai.review.slice(1) : 'Awaiting review in the portal' },
      // Care suggestions only reach the phone once a clinician has approved them.
      ...(ai.suggestions ?? []).map((sg) => ({ label: `Suggested: ${sg.action}`, value: sg.text })),
    ],
  };
}
