import { AppState, Platform } from 'react-native';
import NetInfo from '@react-native-community/netinfo';
import type { Treatment } from '@antigravity-project-spec-pack/domain';
import { applyPull, buildPush, settle, type PullResponse, type PushResponse } from '@antigravity-project-spec-pack/domain/sync';
import { migrateLegacyRecords, setOnLocalChange, useVisitStore, type SyncFailure } from '../store/useVisitStore';
import { api, ApiError, API_URL } from './api';
import { supabase } from './supabase';
import { uuid } from './uuid';

/** Push after edits settle, on coming back online or to the foreground, and every 30 s. */
const DEBOUNCE_MS = 2000;
const EVERY_MS = 30_000;
/** A record or photo that failed this many times shows "Could not sync" (it is still retried). */
export const STALLED_AFTER = 3;

let running = false;
let again = false;
let timer: ReturnType<typeof setTimeout> | null = null;
let deviceId: string | null = null;

const store = () => useVisitStore.getState();

async function getDeviceId(): Promise<string> {
  if (deviceId) return deviceId;
  const { appStorage } = await import('../store/appStorage');
  deviceId = (await appStorage.getItem('device-id')) ?? null;
  if (!deviceId) {
    deviceId = uuid();
    await appStorage.setItem('device-id', deviceId);
  }
  return deviceId;
}

const bump = (failures: Record<string, SyncFailure>, id: string, entity: SyncFailure['entity'], reason: string) => ({
  ...failures,
  [id]: { entity, reason, attempts: (failures[id]?.attempts ?? 0) + 1 },
});

/** Sends the outbox. Records the server refused stay in it and are marked failed. */
async function push(): Promise<void> {
  const s = store();
  const built = buildPush(s, s.outbox, await getDeviceId());
  if (!built) return;
  const res = await api<PushResponse>('/sync/push', { method: 'POST', body: JSON.stringify(built.body) });
  const rejectedIds = new Set(res.rejected.map((r) => r.id));
  // Accepted entries leave the outbox (unless edited again meanwhile); refused ones stay for another try.
  const sent = { ...built.sent };
  for (const e of Object.keys(sent) as (keyof typeof sent)[]) {
    sent[e] = Object.fromEntries(Object.entries(sent[e]).filter(([id]) => !rejectedIds.has(id)));
  }
  const now = store();
  const outbox = settle(now.outbox, sent);
  let failures = { ...now.failures };
  for (const e of Object.keys(sent) as (keyof typeof sent)[]) for (const id of Object.keys(sent[e])) delete failures[id];
  for (const r of res.rejected) failures = bump(failures, r.id, r.entity, r.reason);
  const synced = (id: string) => !outbox.patients[id] && !outbox.cases[id];
  now.replaceSynced({
    patients: now.patients.map((p) => (synced(p.id) ? { ...p, syncState: 'synced' } : p)),
    cases: now.cases.map((c) => (synced(c.id) ? { ...c, syncState: 'synced' } : c)),
    treatments: now.treatments,
    outbox,
  });
  now.setFailures(failures);
}

/** A photo on this device that the server doesn't have yet. The pre photo waits for the assessment (its answers go to the model). */
const photosToSend = (t: Treatment, outboxHas: boolean) => {
  if (outboxHas) return []; // the treatment itself must reach the server first
  const out: { phase: 'pre' | 'post'; uri: string }[] = [];
  if (t.preImageUri && !t.remote?.preStored && t.phase !== 'PRE') out.push({ phase: 'pre', uri: t.preImageUri });
  if (t.postImageUri && !t.remote?.postStored) out.push({ phase: 'post', uri: t.postImageUri });
  return out;
};

async function uploadPhoto(treatmentId: string, phase: 'pre' | 'post', uri: string, measuredLengthCm?: number, takenAt?: string): Promise<void> {
  const form = new FormData();
  if (measuredLengthCm !== undefined) form.append('measured_length_cm', String(measuredLengthCm));
  // When the photo was taken, which can be long before it uploads.
  if (takenAt) form.append('taken_at', takenAt);
  if (Platform.OS === 'web') {
    form.append('photo', await (await fetch(uri)).blob(), `${phase}.jpg`);
  } else {
    // React Native's FormData sends a file from its uri.
    form.append('photo', { uri, name: `${phase}.jpg`, type: 'image/jpeg' } as unknown as Blob);
  }
  await api(`/sync/treatments/${treatmentId}/photos/${phase}`, { method: 'POST', body: form });
}

async function sendPhotos(): Promise<void> {
  const s = store();
  for (const t of s.treatments) {
    for (const { phase, uri } of photosToSend(t, !!s.outbox.treatments[t.id])) {
      const key = `${t.id}:${phase}`;
      try {
        await uploadPhoto(t.id, phase, uri, t.imageMetadata?.measuredLengthCm?.[phase], t.imageMetadata?.takenAt?.[phase]);
        const now = store();
        const failures = { ...now.failures };
        delete failures[key];
        now.setFailures(failures);
        now.replaceSynced({
          patients: now.patients,
          cases: now.cases,
          outbox: now.outbox,
          treatments: now.treatments.map((x) =>
            x.id === t.id ? { ...x, remote: { ...x.remote, ...(phase === 'pre' ? { preStored: true } : { postStored: true }) } } : x,
          ),
        });
      } catch (error) {
        if (error instanceof ApiError && error.status === 0) throw error; // offline: stop and retry later
        store().setFailures(bump(store().failures, key, 'photos', error instanceof Error ? error.message : 'Upload failed.'));
      }
    }
  }
}

/** Brings everything changed elsewhere (the portal, other devices) since the last pull. */
async function pull(): Promise<void> {
  const since = store().since;
  let cursor: string | null = null;
  do {
    const qs: string = `?since=${encodeURIComponent(since ?? '')}${cursor ? `&cursor=${cursor}` : ''}`;
    const res: PullResponse = await api<PullResponse>(`/sync/pull${qs}`);
    const s = store();
    const merged = applyPull(s, res, s.outbox);
    cursor = res.cursor;
    s.replaceSynced({ ...merged.data, outbox: merged.outbox, ...(cursor ? {} : { since: res.serverTime }) });
  } while (cursor);
}

/** One full round: push, photos, pull. Never two at once; a request made meanwhile runs right after. */
export async function syncNow(): Promise<void> {
  if (running) {
    again = true;
    return;
  }
  const { data } = await supabase.auth.getSession();
  if (!data.session || !API_URL) return;
  running = true;
  store().setLastSyncAttempt(new Date().toISOString());
  try {
    await push();
    await sendPhotos();
    await pull();
    store().setLastSyncedAt(new Date().toISOString());
  } catch {
    // Offline or the server is down: everything stays queued for the next round.
  } finally {
    running = false;
    if (again) {
      again = false;
      void syncNow();
    }
  }
}

const soon = () => {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => void syncNow(), DEBOUNCE_MS);
};

let started = false;
export const startSyncEngine = () => {
  if (started) return;
  started = true;
  setOnLocalChange(soon);
  const begin = () => {
    void migrateLegacyRecords().finally(() => void syncNow());
  };
  // The store loads from disk asynchronously; sync only once it has.
  if (useVisitStore.persist.hasHydrated()) begin();
  else useVisitStore.persist.onFinishHydration(begin);

  setInterval(() => void syncNow(), EVERY_MS);
  NetInfo.addEventListener((state) => {
    if (state.isConnected && state.isInternetReachable !== false) void syncNow();
  });
  AppState.addEventListener('change', (state) => {
    if (state === 'active') void syncNow();
  });
};
