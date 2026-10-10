'use client';

import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
  type QueryKey,
} from '@tanstack/react-query';
import type {
  AttentionItem,
  AuditItem,
  CaseCard,
  CaseView,
  Dashboard,
  DraftItem,
  Me,
  Member,
  Page,
  PatientDetail,
  PatientListItem,
  QueueCounts,
  QueueView,
  ReviewView,
  ShareLinkView,
  StatusCounts,
  VisitView,
} from '@antigravity-project-spec-pack/domain/api';
import type { IntakeQuestion } from '@antigravity-project-spec-pack/domain/wound-model';
import { useEffect } from 'react';
import { activeClinicId, api, send } from './api';

/**
 * Every server read in the portal goes through TanStack Query: pages render from cache instantly on revisit,
 * refetch in the background (30 s stale time, on focus), and mutations update or invalidate exactly the keys
 * they change. Keys start with the clinic, so switching clinic never shows another clinic's data.
 */
export type PatientFilters = { q: string; status: '' | 'overdue' | 'review' | 'healing' | 'none'; sort: 'recent' | 'name' | 'nextVisit' | 'lastVisit' };

const c = () => activeClinicId() ?? 'default';
export const keys = {
  me: ['me'] as const,
  all: () => ['clinic', c()] as const,
  dashboard: () => ['clinic', c(), 'dashboard'] as const,
  queueCounts: () => ['clinic', c(), 'queue', 'counts'] as const,
  queue: (view: QueueView) => ['clinic', c(), 'queue', view] as const,
  patients: (f?: PatientFilters) => (f ? (['clinic', c(), 'patients', 'list', f] as const) : (['clinic', c(), 'patients'] as const)),
  patientCounts: () => ['clinic', c(), 'patients', 'counts'] as const,
  patient: (id: string) => ['clinic', c(), 'patient', id] as const,
  case: (id: string) => ['clinic', c(), 'case', id] as const,
  caseVisits: (id: string) => ['clinic', c(), 'case', id, 'visits'] as const,
  visit: (id: string) => ['clinic', c(), 'visit', id] as const,
  shareLinks: (caseId: string) => ['clinic', c(), 'case', caseId, 'share-links'] as const,
  members: () => ['clinic', c(), 'members'] as const,
  audit: (action: string) => ['clinic', c(), 'audit', action] as const,
  intakeQuestions: ['model', 'intake-questions'] as const,
};

const qs = (params: Record<string, string | undefined | null>) => {
  const s = new URLSearchParams(Object.entries(params).filter((e): e is [string, string] => !!e[1])).toString();
  return s ? `?${s}` : '';
};

// ---------- Reads ----------

const ME_KEY = 'wound.me';
const storedMe = (): Me | undefined => {
  try {
    const raw = typeof window !== 'undefined' ? window.localStorage.getItem(ME_KEY) : null;
    return raw ? (JSON.parse(raw) as Me) : undefined;
  } catch {
    return undefined;
  }
};
export const forgetMe = () => {
  try {
    window.localStorage.removeItem(ME_KEY);
  } catch {
    // nothing stored
  }
};

/**
 * The signed-in user and their clinics. The last copy is kept in the browser, so the menu and role checks are
 * right on the first paint; it is refreshed from the server straight away.
 */
export function useMe() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: keys.me, queryFn: () => api<Me>('/me'), staleTime: 5 * 60_000 });
  // After hydration (the prerendered HTML has no user), show the stored copy until the server answers.
  useEffect(() => {
    if (!qc.getQueryData(keys.me)) {
      const stored = storedMe();
      if (stored) qc.setQueryData(keys.me, stored, { updatedAt: 0 });
    }
  }, [qc]);
  useEffect(() => {
    if (!q.data) return;
    try {
      window.localStorage.setItem(ME_KEY, JSON.stringify(q.data));
    } catch {
      // private mode: fine, just slower next time
    }
  }, [q.data]);
  return q;
}

export const useDashboard = (enabled = true) =>
  useQuery({ queryKey: keys.dashboard(), queryFn: () => api<Dashboard>('/dashboard'), enabled });

export const useQueueCounts = (enabled = true) =>
  useQuery({ queryKey: keys.queueCounts(), queryFn: () => api<QueueCounts>('/queue/counts'), enabled, refetchInterval: 60_000 });

export const useQueue = (view: QueueView) =>
  useInfiniteQuery({
    queryKey: keys.queue(view),
    queryFn: ({ pageParam }) => api<Page<DraftItem | AttentionItem>>(`/queue${qs({ view, cursor: pageParam, limit: '25' })}`),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.nextCursor,
    placeholderData: keepPreviousData,
  });

export const fetchPatients = (f: PatientFilters, cursor: string | null) =>
  api<Page<PatientListItem>>(`/patients${qs({ q: f.q.trim() || undefined, status: f.status || undefined, sort: f.sort, cursor, limit: '25' })}`);

export const usePatients = (f: PatientFilters) =>
  useInfiniteQuery({
    queryKey: keys.patients(f),
    queryFn: ({ pageParam }) => fetchPatients(f, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.nextCursor,
    placeholderData: keepPreviousData, // typing in search keeps the old rows on screen until the new ones arrive
  });

export const usePatientCounts = () => useQuery({ queryKey: keys.patientCounts(), queryFn: () => api<StatusCounts>('/patients/counts') });

export const fetchPatient = (id: string) => api<PatientDetail>(`/patients/${id}`);
export const usePatient = (id: string) => useQuery({ queryKey: keys.patient(id), queryFn: () => fetchPatient(id) });

export const fetchCase = (id: string) => api<CaseView>(`/cases/${id}`);
export const useCase = (id: string) =>
  useQuery({
    queryKey: keys.case(id),
    queryFn: () => fetchCase(id),
    // While a visit is being analysed, check every 3 s; otherwise the normal stale time applies.
    refetchInterval: (q) => (q.state.data?.visits.items.some((v) => v.status === 'processing') ? 3000 : false),
  });

/** Older visits than the first page that comes with the case. */
export const useOlderVisits = (caseId: string, after: string | null) =>
  useInfiniteQuery({
    queryKey: keys.caseVisits(caseId),
    queryFn: ({ pageParam }) => api<Page<VisitView>>(`/cases/${caseId}/visits${qs({ cursor: pageParam, limit: '10' })}`),
    initialPageParam: after,
    getNextPageParam: (last) => last.nextCursor,
    enabled: false, // loaded only when asked for
  });

export const useShareLinks = (caseId: string) =>
  useQuery({ queryKey: keys.shareLinks(caseId), queryFn: () => api<ShareLinkView[]>(`/cases/${caseId}/share-links`) });

export const useMembers = (enabled = true) => useQuery({ queryKey: keys.members(), queryFn: () => api<Member[]>('/clinic/members'), enabled });

export const useAudit = (action: string, enabled = true) =>
  useInfiniteQuery({
    enabled,
    queryKey: keys.audit(action),
    queryFn: ({ pageParam }) => api<Page<AuditItem>>(`/audit${qs({ action: action || undefined, cursor: pageParam, limit: '50' })}`),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.nextCursor,
    placeholderData: keepPreviousData,
  });

export const useIntakeQuestions = () =>
  useQuery({ queryKey: keys.intakeQuestions, queryFn: () => api<IntakeQuestion[]>('/model/intake-questions'), staleTime: 60 * 60_000, retry: 1 });

// ---------- Prefetch (hover / focus on a link) ----------

export const prefetchPatient = (qc: QueryClient, id: string) =>
  qc.prefetchQuery({ queryKey: keys.patient(id), queryFn: () => fetchPatient(id), staleTime: 30_000 });
export const prefetchCase = (qc: QueryClient, id: string) =>
  qc.prefetchQuery({ queryKey: keys.case(id), queryFn: () => fetchCase(id), staleTime: 30_000 });

// ---------- Writes ----------

/** After any write the server bumps its clinic cache; the portal drops its lists and summaries the same way. */
const invalidateClinic = (qc: QueryClient, except: QueryKey[] = []) =>
  qc.invalidateQueries({
    queryKey: keys.all(),
    predicate: (q) => !except.some((k) => JSON.stringify(q.queryKey.slice(0, k.length)) === JSON.stringify(k)),
  });

export function useMarkReviewed(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (reviewed: boolean) => send<CaseCard>(`/cases/${caseId}/reviewed`, 'POST', { reviewed }),
    // Optimistic: the case page flips at once and rolls back if the server refuses.
    onMutate: async (reviewed) => {
      await qc.cancelQueries({ queryKey: keys.case(caseId) });
      const before = qc.getQueryData<CaseView>(keys.case(caseId));
      if (before) {
        qc.setQueryData<CaseView>(keys.case(caseId), {
          ...before,
          reviewedAt: reviewed ? new Date().toISOString() : null,
          needsReview: reviewed ? false : before.status === 'overdue' || before.status === 'review',
        });
      }
      return { before };
    },
    onError: (_e, _v, ctx) => ctx?.before && qc.setQueryData(keys.case(caseId), ctx.before),
    onSettled: () => invalidateClinic(qc),
  });
}

export function useReviewVisit(caseId: string, visitId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { decision: 'approved' | 'edited' | 'rejected'; finalReport?: string; reason?: string; corrections?: Record<string, unknown> }) =>
      send<ReviewView>(`/visits/${visitId}/review`, 'POST', body),
    onSuccess: (review) => {
      qc.setQueryData<CaseView>(keys.case(caseId), (cv) =>
        cv ? { ...cv, visits: { ...cv.visits, items: cv.visits.items.map((v) => (v.id === visitId ? { ...v, review } : v)) } } : cv,
      );
      void invalidateClinic(qc, [keys.case(caseId)]);
    },
  });
}

export function useDeleteVisit(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (visitId: string) => send<void>(`/visits/${visitId}`, 'DELETE'),
    onSuccess: () => invalidateClinic(qc),
  });
}

export function useRetryVisit(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (visitId: string) => send<VisitView>(`/visits/${visitId}/retry`, 'POST'),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.case(caseId) }),
  });
}

export function useUpdateCase(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { nextVisit?: string | null; closed?: boolean; remarks?: string | null; woundType?: string }) =>
      send<CaseView>(`/cases/${caseId}`, 'PATCH', body),
    onSuccess: (cv) => {
      qc.setQueryData(keys.case(caseId), cv);
      void invalidateClinic(qc, [keys.case(caseId)]);
    },
  });
}

export function useCreateCase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { patientId: string; location: string; onset: string; woundType?: string; comorbidities?: string[] }) =>
      send<CaseView>('/cases', 'POST', body),
    onSuccess: (cv) => {
      qc.setQueryData(keys.case(cv.id), cv);
      void invalidateClinic(qc, [keys.case(cv.id)]);
    },
  });
}

export function useCreatePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ body, key }: { body: Record<string, unknown>; key: string }) => send<PatientDetail>('/patients', 'POST', body, key),
    onSuccess: (p) => {
      qc.setQueryData(keys.patient(p.id), p);
      void invalidateClinic(qc, [keys.patient(p.id)]);
    },
  });
}

export function useUpdatePatient(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => send<PatientDetail>(`/patients/${id}`, 'PATCH', body),
    onSuccess: (p) => {
      qc.setQueryData(keys.patient(id), p);
      void invalidateClinic(qc, [keys.patient(id)]);
    },
  });
}

export function useDeletePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => send<void>(`/patients/${id}`, 'DELETE'),
    onSuccess: (_r, id) => {
      qc.removeQueries({ queryKey: keys.patient(id) });
      void invalidateClinic(qc);
    },
  });
}

export function useCreateShareLink(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { hidePersonal: boolean; days: number }) => send<ShareLinkView>(`/cases/${caseId}/share-links`, 'POST', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.shareLinks(caseId) }),
  });
}

export function useRevokeShareLink(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => send<ShareLinkView>(`/share-links/${id}/revoke`, 'POST'),
    onSuccess: (link) =>
      qc.setQueryData<ShareLinkView[]>(keys.shareLinks(caseId), (links) => links?.map((l) => (l.id === link.id ? link : l))),
  });
}

export function useInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { email: string; role: string; firstName?: string; lastName?: string }) => send<Member>('/clinic/members', 'POST', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.members() }),
  });
}

export function useUpdateMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; role?: string; active?: boolean }) => send<Member>(`/clinic/members/${id}`, 'PATCH', body),
    onMutate: async ({ id, ...body }) => {
      await qc.cancelQueries({ queryKey: keys.members() });
      const before = qc.getQueryData<Member[]>(keys.members());
      qc.setQueryData<Member[]>(keys.members(), (ms) => ms?.map((m) => (m.id === id ? ({ ...m, ...body } as Member) : m)));
      return { before };
    },
    onError: (_e, _v, ctx) => ctx?.before && qc.setQueryData(keys.members(), ctx.before),
    onSettled: () => qc.invalidateQueries({ queryKey: keys.members() }),
  });
}

/** The signed-in user's role in the active clinic (undefined while loading). */
export function useRole() {
  const me = useMe();
  const memberships = me.data?.memberships ?? [];
  return (memberships.find((m) => m.clinicId === activeClinicId()) ?? memberships[0])?.role;
}
