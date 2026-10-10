# Deploying

Four services, all on free tiers to start. Deploy them in this order, because each one needs the address of the
one before.

| Service | Code | Host | Region | Config in the repo |
| --- | --- | --- | --- | --- |
| Database, sign-in, photos | `prisma/` | Supabase | Seoul (ap-northeast-2) | `prisma/migrations` |
| Redis (cache, rate limits, jobs) | — | Upstash | Seoul / ap-northeast | — |
| Wound model API | `wound-ai/` | Hugging Face Space (Docker) | (Hugging Face's) | `wound-ai/Dockerfile` |
| NestJS API + background jobs | `apps/api` | Render (web service) | Singapore | `render.yaml` |
| Web portal | `apps/web` | Vercel | Seoul (`icn1`) | `vercel.json` |

Keep the API, the database and Redis close together: almost every API request is one database round trip and a
Redis read, so the distance between them is most of a request's time. When the India-region database exists,
move all of them to Mumbai together (Render has no Mumbai region: use a host that does, or Singapore).

Everything here is a **research prototype, not for patient care**: no real patient photos until the ethics and
regulatory steps in `wound-ai/docs/roadmap.md` are done, and patient data must move to an India-region database
before then.

## 0. Supabase

The project is set up; the migrations create every table with row-level security on and no client access (all
data goes through the API).

1. Authentication → Providers → Email: **turn sign-ups off** (accounts are invite-only).
2. Authentication → URL Configuration: add the Vercel URL and `<vercel-url>/login` to the redirect URLs (invite
   and password-reset emails link there).
3. Project settings → Data API: turn it **off** (nothing uses it; the lockdown already denies it).
4. Copy two connection strings: the **transaction pooler** (port 6543) → `DATABASE_URL`, the **session pooler**
   (port 5432) → `DIRECT_URL`.

## 1. Redis: Upstash

1. Create a Redis database (free), region closest to Seoul.
2. Copy its `rediss://` URL → `REDIS_URL`.

The free plan has a monthly command allowance. The API is tuned for it: idle workers wait 30 s per Redis call
(`QUEUE_IDLE_SECONDS`), and the audit buffer is only read after something was logged. Without Redis the API
still works (no cache, jobs run inline), so an outage degrades speed, not function.

## 2. Model: Hugging Face Space

1. Create a **private model repo** (e.g. `you/wound-models`) and upload the trained `*.pt` files (`boundary.pt`,
   `wound_type.pt`). They are never committed to git.
2. Create a **Space**: SDK *Docker*, hardware *CPU basic*, visibility private if your plan allows.
3. Push the contents of `wound-ai/` to the Space repo, and add this at the top of the Space's `README.md`:

   ```yaml
   ---
   title: Wound model API
   sdk: docker
   app_port: 7860
   ---
   ```

4. Space **secrets**: `WOUND_API_KEY` (a long random string), `HF_MODEL_REPO` (`you/wound-models`) and `HF_TOKEN`
   (a read token for that repo).
5. Check: `https://<space>.hf.space/health` lists the model versions. The Space sleeps when idle; visit analysis
   runs in the background and is retried, so a sleeping model only delays a draft.

## 3. API: Render

1. Render → **New → Blueprint** → this GitHub repo. It reads `render.yaml` (Singapore, health check on
   `/api/health/ready`, migrations applied on every deploy).
2. Fill in the secrets it asks for (listed in `render.yaml` and `.env.example`): the Supabase values,
   `REDIS_URL`, `WOUND_API_URL` / `WOUND_API_KEY`, `SHARE_LINK_SECRET` (new random string), `PUBLIC_API_URL`
   (`https://<service>.onrender.com/api`), and `PORTAL_URL` / `CORS_ORIGINS` (the Vercel URL, after step 4).
3. Check: `https://<service>.onrender.com/api/health/ready` shows database, Redis and model.
4. Make the first admin: `node --env-file=.env tools/bootstrap-clinic.mjs you@example.com "Clinic name"` (the
   account must exist in Supabase → Authentication). Everyone else is invited from the portal (Users & roles).

The free plan sleeps after 15 minutes idle (the first request then takes ~30 s) and runs the background jobs in
the same process. To scale: a paid plan, more instances with `RUN_WORKERS=0`, and the worker service sketched at
the bottom of `render.yaml` (`WORKER_ONLY=1`). Instances share nothing in memory, so any number can run.

## 1–3 on Railway instead (Redis + model + API in one project)

One Railway project (`wound`, region Singapore) replaces Upstash, the Hugging Face Space and Render. The two
`.railwayignore` files keep the uploads small. **The model needs the Hobby plan or higher:** on the trial plan
(1 GB memory per service, 500 MB volumes) it loads four models but is killed on the first photo, and the five weight
files (506 MB) don't fit the volume. The API and Redis run fine on the trial.

```bash
railway init --name wound
railway add --database redis
railway add --service wound-model --variables "PORT=7860"            # + WOUND_API_KEY (a long random string)
railway service link wound-model && railway volume add --mount-path /app/checkpoints
railway up ./wound-ai --path-as-root --service wound-model           # first deploy, then the weights:
railway ssh keys add                                                 # once: file uploads go over SSH
for f in boundary wound_type tissue burn_depth dfu_wagner; do
  railway volume files --volume wound-model-volume upload wound-ai/checkpoints/$f.pt /$f.pt --overwrite; done
railway service restart --service wound-model --yes
railway add --service wound-api && railway domain --service wound-api
railway up --service wound-api
# every service in one region (single-region plans fail a deploy while two are listed):
for s in Redis wound-model wound-api; do railway service scale --service $s southeast-asia=1 sfo=0; done
```

- **Service settings** (set once with `railway api 'mutation { serviceInstanceUpdate(...) }'` or in the dashboard; this
  CLI no longer applies `railway.json`). API: build `npx prisma generate && npx nx build api`, pre-deploy
  `npx prisma migrate deploy`, start `node dist/apps/api/main.js`, health check `/api/health/ready`. Model: health
  check `/health` (it builds from `wound-ai/Dockerfile`).
- **Weights** live on the model service's volume (`/app/checkpoints`), not in git or the image: after training, upload
  the new `.pt` and restart the service. `HF_MODEL_REPO` stays unset.
- **The model has no public address.** The API reaches it on the private network:
  `WOUND_API_URL=http://${{wound-model.RAILWAY_PRIVATE_DOMAIN}}:7860`, `WOUND_API_KEY=${{wound-model.WOUND_API_KEY}}`.
- **API variables:** the Render list below, with `REDIS_URL=${{Redis.REDIS_URL}}`,
  `PUBLIC_API_URL=https://${{RAILWAY_PUBLIC_DOMAIN}}/api`, plus `RAILPACK_NODE_VERSION=24` and `NPM_CONFIG_INCLUDE=dev`
  (the build needs nx and prisma, which are dev dependencies).
- `railway up` deploys from this computer. For deploys on every push:
  `railway service source connect --repo opalvedansh/wound_1 --branch main --service wound-api` (the model service also
  needs its root directory set to `wound-ai`).

## 4. Web portal: Vercel

1. Vercel → **New Project** → this repo → Root Directory **`./`** (the repo root). `vercel.json` sets the build
   and the Seoul region.
2. Environment variables: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, and
   `NEXT_PUBLIC_API_URL` = `https://<service>.onrender.com/api`.
3. `NEXT_PUBLIC_*` values are fixed at build time: redeploy after changing them.
4. Never add `WOUND_API_KEY`, `DATABASE_URL` or the Supabase secret key here (the portal needs none of them).

## 5. Mobile app

`apps/mobile/.env`: `EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY` and `EXPO_PUBLIC_API_URL` (the
Render URL). Build with EAS (`apps/mobile/eas.json`). The app works offline and syncs when it can; members sign in
with the same invited accounts as the portal.

## After deploying

- Add the Vercel URL to Render's `CORS_ORIGINS` and `PORTAL_URL`, and to Supabase's redirect URLs.
- Run the checks against the deployed services: `API_URL=https://<service>.onrender.com/api node --env-file=.env
  tools/e2e-check.mjs` (it makes and removes its own throwaway clinic), and
  `PORTAL_URL=<vercel-url> … tools/portal-check.mjs`.
- Sign in on a phone, run one visit with the printed sticker, and approve the draft in the portal.

## Scaling past the free tiers

What changes, in order, as clinics and users grow (no code changes needed):

1. Render paid instances (no sleep), then 2+ web instances with `RUN_WORKERS=0` plus the worker service.
2. Supabase Pro (more connections and storage); raise `DB_POOL_MAX` so instances × pool stays under the pooler's
   limit.
3. Upstash pay-as-you-go (no command cap).
4. The model on a GPU host or more Space replicas; the job queue already spreads analyses across workers.

Measured capacity and query plans: [`docs/performance.md`](performance.md).
