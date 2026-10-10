import type { AnalyzeResponse, Care, IntakeAnswers, Progress, ReviewDecision } from '@antigravity-project-spec-pack/domain/wound-model';
import type { CaseCard, Consent, PatientListItem, ReviewView, VisitStatus, VisitView, WoundStatus } from '@antigravity-project-spec-pack/domain/api';

/** Database rows as the JSON clients read: ISO timestamps, YYYY-MM-DD calendar days. */

export const iso = (d: Date | null | undefined) => (d ? d.toISOString() : null);
export const day = (d: Date | null | undefined) => (d ? d.toISOString().slice(0, 10) : null);

export const areaChangePct = (first: number | null, latest: number | null) =>
  first && latest !== null && first > 0 ? Math.round(((latest - first) / first) * 1000) / 10 : null;

/** A case shows in the review queue while it is overdue or needs review and nobody has reviewed it since its last visit. */
export const needsReview = (c: { status: string | null; reviewedAt: Date | null; lastVisitAt: Date | null }) =>
  (c.status === 'overdue' || c.status === 'review') && (!c.reviewedAt || (!!c.lastVisitAt && c.reviewedAt < c.lastVisitAt));

export interface PatientRow {
  id: string;
  patientId: string;
  firstName: string;
  lastName: string;
  sex: string;
  dateOfBirth: Date | null;
  ageYears: number | null;
  status: string | null;
  lastVisitAt: Date | null;
  nextVisitDue: Date | null;
  createdAt: Date;
  cases?: { woundType: string; location: string }[];
  _count?: { cases: number };
}

export const toPatientListItem = (p: PatientRow): PatientListItem => {
  const first = p.cases?.[0];
  return {
    id: p.id,
    patientId: p.patientId,
    firstName: p.firstName,
    lastName: p.lastName,
    sex: p.sex,
    dateOfBirth: day(p.dateOfBirth),
    ageYears: p.ageYears,
    status: p.status as WoundStatus | null,
    lastVisitAt: iso(p.lastVisitAt),
    nextVisitDue: day(p.nextVisitDue),
    openWounds: p._count?.cases ?? 0,
    woundSummary: first ? [first.woundType === 'Not recorded' ? null : first.woundType, first.location].filter(Boolean).join(' · ') : null,
    createdAt: p.createdAt.toISOString(),
  };
};

export interface CaseRow {
  id: string;
  location: string;
  woundType: string;
  onset: Date;
  status: string | null;
  statusReason: string | null;
  visitCount: number;
  lastVisitAt: Date | null;
  nextVisitDue: Date | null;
  firstAreaCm2: number | null;
  latestAreaCm2: number | null;
  closedAt: Date | null;
  reviewedAt: Date | null;
}

export const toCaseCard = (c: CaseRow, thumbUrl: string | null = null, outline: CaseCard['outline'] = null): CaseCard => ({
  id: c.id,
  location: c.location,
  woundType: c.woundType,
  onset: day(c.onset) as string,
  status: c.status as WoundStatus | null,
  statusReason: c.statusReason,
  visitCount: c.visitCount,
  lastVisitAt: iso(c.lastVisitAt),
  nextVisitDue: day(c.nextVisitDue),
  firstAreaCm2: c.firstAreaCm2,
  latestAreaCm2: c.latestAreaCm2,
  areaChangePct: areaChangePct(c.firstAreaCm2, c.latestAreaCm2),
  closedAt: iso(c.closedAt),
  reviewedAt: iso(c.reviewedAt),
  needsReview: needsReview(c),
  thumbUrl,
  outline,
});

export const toConsent = (value: unknown): Consent | null => (value && typeof value === 'object' ? (value as Consent) : null);

export interface ReviewRow {
  decision: string;
  finalReport: string | null;
  reason: string | null;
  reviewerId: string;
  createdAt: Date;
}

export const toReviewView = (r: ReviewRow): ReviewView => ({
  decision: r.decision as ReviewDecision,
  finalReport: r.finalReport,
  reason: r.reason,
  reviewerId: r.reviewerId,
  createdAt: r.createdAt.toISOString(),
});

interface ImageRow {
  imageUrl: string;
  thumbPath: string | null;
}

export interface VisitRow {
  id: string;
  status: string;
  error: string | null;
  findings: unknown;
  intake: unknown;
  draftReport: string | null;
  progress: unknown;
  care: unknown;
  rulesVersion: string | null;
  createdAt: Date;
  review?: ReviewRow | null;
  phase: {
    treatment: { id: string; sequence: number; phases: { image: ImageRow | null; aiResult: { status: string; findings: unknown } | null }[] };
    image: ImageRow | null;
  };
}

/**
 * A visit is a treatment's PRE result: the one a clinician reviews. The POST result (same visit, after cleaning)
 * is shown inside it, so every list of visits filters on this.
 */
export const PRE_VISIT = { phase: { phaseType: 'PRE' } } as const;

export const toVisitView = (r: VisitRow, urls: Map<string, string | null>): VisitView => ({
  id: r.id,
  treatmentId: r.phase.treatment.id,
  sequence: r.phase.treatment.sequence,
  status: r.status as VisitStatus,
  error: r.error,
  takenAt: r.createdAt.toISOString(),
  photoUrl: r.phase.image ? urls.get(r.phase.image.imageUrl) ?? null : null,
  thumbUrl: r.phase.image?.thumbPath ? urls.get(r.phase.image.thumbPath) ?? null : null,
  findings: (r.findings as AnalyzeResponse | null) ?? null,
  intake: (r.intake ?? {}) as IntakeAnswers,
  draftReport: r.draftReport,
  review: r.review ? toReviewView(r.review) : null,
  post: postOf(r, urls),
  progress: (r.progress as Progress | null) ?? null,
  care: (r.care as Care | null) ?? null,
  rulesVersion: r.rulesVersion,
});

const postOf = (r: VisitRow, urls: Map<string, string | null>): VisitView['post'] => {
  const post = r.phase.treatment.phases[0];
  if (!post?.image) return null;
  return {
    status: (post.aiResult?.status ?? 'processing') as VisitStatus,
    photoUrl: urls.get(post.image.imageUrl) ?? null,
    findings: (post.aiResult?.findings as AnalyzeResponse | null) ?? null,
  };
};

/** Storage paths to sign for a set of visits (photo and thumbnail). */
export const visitPaths = (rows: VisitRow[]) =>
  rows
    .flatMap((r) => [r.phase.image, r.phase.treatment.phases[0]?.image ?? null])
    .flatMap((img) => (img ? [img.imageUrl, img.thumbPath ?? ''] : []))
    .filter(Boolean);

export const visitSelect = {
  id: true,
  status: true,
  error: true,
  findings: true,
  intake: true,
  draftReport: true,
  progress: true,
  care: true,
  rulesVersion: true,
  createdAt: true,
  review: true,
  phase: {
    select: {
      treatment: {
        select: {
          id: true,
          sequence: true,
          phases: {
            where: { phaseType: 'POST', deletedAt: null },
            select: { image: { select: { imageUrl: true, thumbPath: true } }, aiResult: { select: { status: true, findings: true } } },
          },
        },
      },
      image: { select: { imageUrl: true, thumbPath: true } },
    },
  },
} as const;
