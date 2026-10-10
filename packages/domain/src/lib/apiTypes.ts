/**
 * The JSON the NestJS API returns, shared by the API, the web portal and the mobile app.
 * Import through `@antigravity-project-spec-pack/domain/api` (the package index pulls in a browser client).
 * Dates are ISO strings; calendar days are YYYY-MM-DD.
 */
import type { AnalyzeResponse, Care, IntakeAnswers, Progress, ReviewDecision } from './woundModel';

export type ClinicRole = 'ADMIN' | 'DOCTOR' | 'FRONT_DESK';
export type WoundStatus = 'healing' | 'review' | 'overdue';

export interface Page<T> {
  items: T[];
  /** Pass back as `cursor` for the next page; null on the last page. */
  nextCursor: string | null;
}

export interface Me {
  id: string;
  email: string | null;
  firstName: string;
  lastName: string;
  /** May edit the shared question catalogue. */
  platformAdmin: boolean;
  memberships: { clinicId: string; clinicName: string; role: ClinicRole }[];
}

export interface PatientListItem {
  id: string;
  patientId: string;
  firstName: string;
  lastName: string;
  sex: string;
  dateOfBirth: string | null;
  ageYears: number | null;
  status: WoundStatus | null;
  lastVisitAt: string | null;
  nextVisitDue: string | null;
  openWounds: number;
  /** "Diabetic foot ulcer · Left heel", the first open wound. */
  woundSummary: string | null;
  createdAt: string;
}

export interface StatusCounts {
  all: number;
  overdue: number;
  review: number;
  healing: number;
}

export interface Consent {
  care: boolean;
  photos?: boolean;
  location?: boolean;
  aiTraining: boolean;
  noticeVersion: string;
  recordedAt: string;
}

export interface CaseCard {
  id: string;
  location: string;
  woundType: string;
  onset: string;
  status: WoundStatus | null;
  statusReason: string | null;
  visitCount: number;
  lastVisitAt: string | null;
  nextVisitDue: string | null;
  firstAreaCm2: number | null;
  latestAreaCm2: number | null;
  /** % change of area since the first measured visit; negative = shrinking. */
  areaChangePct: number | null;
  closedAt: string | null;
  reviewedAt: string | null;
  needsReview: boolean;
  thumbUrl: string | null;
  outline: [number, number][][] | null;
}

export interface PatientDetail extends PatientListItem {
  mobile: string | null;
  location: string | null;
  referral: string | null;
  notes: string | null;
  consent: Consent | null;
  /** Hidden from the front desk (no clinical data). */
  cases: CaseCard[] | null;
}

export type VisitStatus = 'processing' | 'ok' | 'retake' | 'no_wound_found' | 'failed';

export interface ReviewView {
  decision: ReviewDecision;
  finalReport: string | null;
  reason: string | null;
  reviewerId: string;
  createdAt: string;
}

export interface VisitView {
  id: string;
  treatmentId: string;
  sequence: number;
  status: VisitStatus;
  error: string | null;
  takenAt: string;
  photoUrl: string | null;
  thumbUrl: string | null;
  findings: AnalyzeResponse | null;
  intake: IntakeAnswers;
  draftReport: string | null;
  review: ReviewView | null;
  /** The same visit's photo after cleaning, before the dressing; null if none was taken. */
  post: { status: VisitStatus; photoUrl: string | null; findings: AnalyzeResponse | null; takenAt: string } | null;
  /** The wound's depth the clinician probed at this visit; null if not recorded. */
  depthCm: number | null;
  /** Healing and care suggestions, once the treatment report has run. */
  progress: Progress | null;
  care: Care | null;
  rulesVersion: string | null;
}

export interface CaseView extends CaseCard {
  comorbidities: string[];
  remarks: string | null;
  patient: { id: string; firstName: string; lastName: string; patientId: string };
  /** Area of every measured visit, oldest first, for the chart. */
  areaSeries: { at: string; areaCm2: number; sequence: number }[];
  /** Newest first; more via GET /cases/:id/visits?cursor=… */
  visits: Page<VisitView>;
}

export interface Dashboard {
  patients: number;
  openWounds: number;
  shrinking: number;
  visitsThisWeek: number;
  visitsLastWeek: number;
  draftsToReview: number;
  urgentDrafts: number;
  needsAttention: number;
  visitsPerWeek: { weekStart: string; count: number }[];
  woundTypes: { label: string; count: number }[];
  healing: { caseId: string; patient: string; location: string; changePct: number }[];
  attention: AttentionItem[];
  upcoming: { caseId: string; patientId: string; patient: string; location: string; due: string }[];
}

export interface AttentionItem {
  caseId: string;
  patientId: string;
  patient: string;
  location: string;
  status: WoundStatus;
  reason: string | null;
  lastVisitAt: string | null;
  nextVisitDue: string | null;
  reviewedAt: string | null;
  thumbUrl: string | null;
}

export interface DraftItem {
  visitId: string;
  caseId: string;
  patient: string;
  location: string;
  takenAt: string;
  urgent: boolean;
  flagCount: number;
  areaCm2: number | null;
  thumbUrl: string | null;
}

export type QueueView = 'drafts' | 'attention' | 'overdue' | 'review' | 'reviewed';

export interface QueueCounts {
  drafts: number;
  attention: number;
  overdue: number;
  review: number;
  reviewed: number;
}

export interface Member {
  id: string;
  userId: string;
  email: string;
  name: string;
  role: ClinicRole;
  active: boolean;
  createdAt: string;
}

export interface AuditItem {
  id: string;
  at: string;
  action: string;
  entity: string | null;
  entityId: string | null;
  user: string | null;
  details: Record<string, unknown> | null;
}

export interface ShareLinkView {
  id: string;
  hidePersonal: boolean;
  createdAt: string;
  expiresAt: string;
  revokedAt: string | null;
  viewCount: number;
  state: 'active' | 'expired' | 'revoked';
  /** Only in the response that creates the link; never stored or shown again. */
  url?: string;
}
