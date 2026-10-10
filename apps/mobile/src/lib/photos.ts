import type { Treatment } from '@antigravity-project-spec-pack/domain';
import { woundTypeName } from '@antigravity-project-spec-pack/domain/wound-model';

/** The photo to show: the copy on this device, else the server's (taken on another device or before a reinstall). */
export const prePhoto = (t: Treatment | undefined) => t?.preImageUri ?? t?.remote?.preUrl ?? undefined;
export const postPhoto = (t: Treatment | undefined) => t?.postImageUri ?? t?.remote?.postUrl ?? undefined;

/** The AI draft for a treatment's pre-treatment photo, in words. */
const TRAJECTORY = { improving: 'Improving', static: 'Static', deteriorating: 'Deteriorating' } as const;
const POST_STATUS = {
  processing: 'Analysing…',
  ok: 'Analysed',
  retake: 'Kept, but the model could not use it',
  no_wound_found: 'Kept, no wound found in it',
  failed: 'Analysis failed',
} as const;

export function aiSummary(t: Treatment): { label: string; value: string }[] | null {
  const ai = t.remote?.ai;
  if (!ai) {
    if (t.preImageUri && !t.remote?.preStored) return [{ label: 'AI analysis', value: t.phase === 'PRE' ? 'Starts after the assessment is saved' : 'Waiting to upload the photo' }];
    return null;
  }
  if (ai.status === 'processing') return [{ label: 'AI analysis', value: 'Analysing the photo…' }];
  if (ai.status === 'failed') return [{ label: 'AI analysis', value: 'Failed. A doctor can retry it from the portal.' }];
  if (ai.status === 'retake') return [{ label: 'AI analysis', value: 'The photo could not be used (blank, too small, or too dark or bright). Take it again.' }];
  if (ai.status === 'no_wound_found') return [{ label: 'AI analysis', value: 'No wound found in the photo.' }];
  const reviewed = ai.review === 'approved' || ai.review === 'edited';
  const since = ai.areaReductionSinceFirstPct;
  return [
    { label: 'Area (AI)', value: ai.areaCm2 !== null ? `${ai.areaCm2} cm²` : 'Not measured (no sticker found)' },
    { label: 'Wound type (AI)', value: ai.woundType ? woundTypeName(ai.woundType) : '—' },
    ...(ai.trajectory
      ? [{ label: 'Healing (AI)', value: `${TRAJECTORY[ai.trajectory]}${reviewed ? '' : ', awaiting review'}` }]
      : []),
    ...(since !== null && since !== undefined
      ? [{ label: 'Since first visit (AI)', value: `${Math.abs(since)}% ${since >= 0 ? 'smaller' : 'larger'}` }]
      : []),
    ...(ai.postStatus ? [{ label: 'After-cleaning photo', value: POST_STATUS[ai.postStatus] }] : []),
    ...(ai.urgent ? [{ label: 'Flag', value: 'Urgent: see the draft in the portal' }] : []),
    { label: 'Clinician review', value: ai.review ? ai.review[0].toUpperCase() + ai.review.slice(1) : 'Awaiting review in the portal' },
    // Care suggestions only reach the phone once a clinician has approved them.
    ...(ai.suggestions ?? []).map((sg) => ({ label: `Suggested: ${sg.action}`, value: sg.text })),
  ];
}
