import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { Case, Patient, Treatment } from '@antigravity-project-spec-pack/domain';
import { emptyOutbox, track, type Outbox, type SyncEntity } from '@antigravity-project-spec-pack/domain/sync';
import { uuid } from '../lib/uuid';
import { secureStorage } from './secureStorage';
import { appStorage } from './appStorage';

/** A record the server refused, or one that kept failing: shown as "Could not sync" until it goes through. */
export interface SyncFailure {
  entity: SyncEntity | 'photos';
  reason: string;
  attempts: number;
}

interface VisitState {
  patients: Patient[];
  cases: Case[];
  treatments: Treatment[];
  /** Fields changed on this device and not yet pushed. */
  outbox: Outbox;
  failures: Record<string, SyncFailure>;
  /** serverTime of the last complete pull. */
  since: string | null;
  lastSyncAttempt?: string;
  lastSyncedAt?: string;

  addPatient: (patient: Omit<Patient, 'id' | 'syncState'>) => string;
  addCase: (caseData: Omit<Case, 'id' | 'syncState' | 'createdAt'>) => string;
  addTreatment: (caseId: string) => string;
  updateTreatment: (treatmentId: string, updates: Partial<Treatment>) => void;
  completePrePhase: (treatmentId: string) => void;
  completePostPhase: (treatmentId: string) => void;

  /** Sync engine only. */
  replaceSynced: (next: Pick<VisitState, 'patients' | 'cases' | 'treatments' | 'outbox'> & { since?: string | null }) => void;
  setFailures: (failures: Record<string, SyncFailure>) => void;
  setLastSyncAttempt: (date: string) => void;
  setLastSyncedAt: (date: string) => void;
  /** Everything on this device, removed at sign-out. */
  clearAll: () => void;

  // UI flow flags (not persisted)
  imageCaptured: boolean;
  clinicalDataSaved: boolean;
  aiProcessed: boolean;
  resetVisit: () => void;
  setImageCaptured: (val: boolean) => void;
  setClinicalDataSaved: (val: boolean) => void;
  setAiProcessed: (val: boolean) => void;
}

/** Notified after every local edit, so the sync engine can push soon after (set by syncManager). */
let onLocalChange: () => void = () => undefined;
export const setOnLocalChange = (fn: () => void) => {
  onLocalChange = fn;
};

const EMPTY = { patients: [], cases: [], treatments: [], outbox: emptyOutbox(), failures: {}, since: null };

export const useVisitStore = create<VisitState>()(
  persist(
    (set, get) => {
      const edit = (entity: SyncEntity, id: string, fields: string[], update: (s: VisitState) => Partial<VisitState>) => {
        set((s) => ({ ...update(s), outbox: track(s.outbox, entity, id, fields) }));
        onLocalChange();
      };
      return {
        ...EMPTY,

        addPatient: (data) => {
          const id = uuid();
          edit('patients', id, Object.keys(data), (s) => ({ patients: [...s.patients, { ...data, id, syncState: 'pending' }] }));
          return id;
        },

        addCase: (data) => {
          const id = uuid();
          const record: Case = { ...data, id, syncState: 'pending', createdAt: new Date().toISOString() };
          edit('cases', id, Object.keys(record), (s) => ({ cases: [...s.cases, record] }));
          return id;
        },

        addTreatment: (caseId) => {
          const id = uuid();
          const caseTreatments = get().treatments.filter((t) => t.caseId === caseId);
          const previous = caseTreatments.length > 0 ? caseTreatments[caseTreatments.length - 1] : null;
          const record: Treatment = {
            id,
            caseId,
            sequenceNumber: Math.max(0, ...caseTreatments.map((t) => t.sequenceNumber)) + 1,
            phase: 'PRE',
            // Every visit takes its own pre-treatment photo: last visit's photo, carried over, would make the wound
            // look unchanged. The camera shows it as a framing guide instead (CameraScreen).
            assessment: previous?.assessment ? { ...previous.assessment } : undefined,
            createdAt: new Date().toISOString(),
          };
          edit('treatments', id, Object.keys(record), (s) => ({ treatments: [...s.treatments, record] }));
          return id;
        },

        updateTreatment: (treatmentId, updates) => {
          // A new photo replaces what the server had for that phase, so it is uploaded again.
          const photoChanged = 'preImageUri' in updates || 'postImageUri' in updates;
          edit('treatments', treatmentId, Object.keys(updates), (s) => ({
            treatments: s.treatments.map((t) =>
              t.id === treatmentId
                ? {
                    ...t,
                    ...updates,
                    ...(photoChanged && t.remote
                      ? { remote: { ...t.remote, ...('preImageUri' in updates ? { preStored: false } : {}), ...('postImageUri' in updates ? { postStored: false } : {}) } }
                      : {}),
                  }
                : t,
            ),
          }));
        },

        completePrePhase: (treatmentId) =>
          edit('treatments', treatmentId, ['phase'], (s) => ({ treatments: s.treatments.map((t) => (t.id === treatmentId ? { ...t, phase: 'POST' } : t)) })),

        completePostPhase: (treatmentId) =>
          edit('treatments', treatmentId, ['phase'], (s) => ({ treatments: s.treatments.map((t) => (t.id === treatmentId ? { ...t, phase: 'COMPLETED' } : t)) })),

        replaceSynced: ({ since, ...next }) => set((s) => ({ ...next, since: since === undefined ? s.since : since })),
        setFailures: (failures) => set({ failures }),
        setLastSyncAttempt: (date) => set({ lastSyncAttempt: date }),
        setLastSyncedAt: (date) => set({ lastSyncedAt: date }),
        clearAll: () => set({ ...EMPTY, outbox: emptyOutbox(), lastSyncAttempt: undefined, lastSyncedAt: undefined }),

        imageCaptured: false,
        clinicalDataSaved: false,
        aiProcessed: false,
        resetVisit: () => set({ imageCaptured: false, clinicalDataSaved: false, aiProcessed: false }),
        setImageCaptured: (val) => set({ imageCaptured: val }),
        setClinicalDataSaved: (val) => set({ clinicalDataSaved: val }),
        setAiProcessed: (val) => set({ aiProcessed: val }),
      };
    },
    {
      // v3: records synced through the API, kept in SQLite (the keychain is too small for them).
      name: 'visit-storage-v3',
      storage: createJSONStorage(() => appStorage),
      partialize: (s) => ({
        patients: s.patients,
        cases: s.cases,
        treatments: s.treatments,
        outbox: s.outbox,
        failures: s.failures,
        since: s.since,
        lastSyncAttempt: s.lastSyncAttempt,
        lastSyncedAt: s.lastSyncedAt,
      }),
    },
  ),
);

/** Records waiting to be sent, or failing. */
export const pendingIds = (outbox: Outbox) => new Set([...Object.keys(outbox.patients), ...Object.keys(outbox.cases), ...Object.keys(outbox.treatments)]);

/**
 * Records from the previous version (kept in the keychain, with short ids the server can't take) move into
 * this store with new ids, queued for upload. Runs once; the old copy is removed after it moved.
 */
export async function migrateLegacyRecords(): Promise<void> {
  const raw = await Promise.resolve(secureStorage.getItem('visit-storage-v2')).catch(() => null);
  await Promise.resolve(secureStorage.removeItem('visit-storage')).catch(() => undefined);
  if (!raw) return;
  try {
    const old = (JSON.parse(raw) as { state?: { patients?: Patient[]; cases?: Case[]; treatments?: Treatment[] } }).state ?? {};
    const ids = new Map<string, string>();
    const idFor = (oldId: string) => ids.get(oldId) ?? (ids.set(oldId, uuid()), ids.get(oldId) as string);
    const patients = (old.patients ?? []).map((p) => ({ ...p, id: idFor(p.id), syncState: 'pending' as const }));
    const cases = (old.cases ?? []).filter((c) => ids.has(c.patientId)).map((c) => ({ ...c, id: idFor(c.id), patientId: idFor(c.patientId), syncState: 'pending' as const }));
    const treatments = (old.treatments ?? []).filter((t) => ids.has(t.caseId)).map((t) => ({ ...t, id: idFor(t.id), caseId: idFor(t.caseId) }));
    useVisitStore.setState((s) => {
      let outbox = s.outbox;
      for (const p of patients) outbox = track(outbox, 'patients', p.id, Object.keys(p));
      for (const c of cases) outbox = track(outbox, 'cases', c.id, Object.keys(c));
      for (const t of treatments) outbox = track(outbox, 'treatments', t.id, Object.keys(t));
      return { patients: [...s.patients, ...patients], cases: [...s.cases, ...cases], treatments: [...s.treatments, ...treatments], outbox };
    });
    await Promise.resolve(secureStorage.removeItem('visit-storage-v2'));
  } catch {
    // Unreadable: leave it where it is rather than lose it.
  }
}
