import { createHmac, randomBytes } from 'node:crypto';
import { GoneException, Injectable, NotFoundException } from '@nestjs/common';
import { z } from 'zod';
import { woundTypeName, type AnalyzeResponse } from '@antigravity-project-spec-pack/domain/wound-model';
import type { ShareLinkView } from '@antigravity-project-spec-pack/domain/api';
import type { ClinicContext } from '../auth/clinic.guard';
import { AuditService } from '../platform/audit.service';
import { parse } from '../platform/validation';
import { PrismaService } from '../prisma.service';
import { StorageService } from '../visits/storage.service';

const DAY_MS = 86_400_000;
export const createShare = z.object({ hidePersonal: z.boolean().default(true), days: z.coerce.number().int().min(1).max(30).default(7) });

const esc = (s: unknown) =>
  String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string);
const fmt = (d: Date) => d.toISOString().slice(0, 10);

/**
 * Expiring, revocable, read-only links to one wound's report, for a referring doctor or the patient.
 * The token is shown once; only an HMAC of it is stored, so a database leak reveals no working links.
 */
@Injectable()
export class ShareService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
    private readonly audit: AuditService,
  ) {}

  private hash(token: string) {
    const secret = process.env['SHARE_LINK_SECRET'] || process.env['SUPABASE_SECRET_KEY'] || 'dev-only-share-secret';
    return createHmac('sha256', secret).update(`share:${token}`).digest('hex');
  }

  private view(l: { id: string; hidePersonal: boolean; createdAt: Date; expiresAt: Date; revokedAt: Date | null; viewCount: number }): ShareLinkView {
    return {
      id: l.id,
      hidePersonal: l.hidePersonal,
      createdAt: l.createdAt.toISOString(),
      expiresAt: l.expiresAt.toISOString(),
      revokedAt: l.revokedAt?.toISOString() ?? null,
      viewCount: l.viewCount,
      state: l.revokedAt ? 'revoked' : l.expiresAt < new Date() ? 'expired' : 'active',
    };
  }

  async list(ctx: ClinicContext, caseId: string): Promise<ShareLinkView[]> {
    const rows = await this.prisma.shareLink.findMany({ where: { caseId, clinicId: ctx.clinicId }, orderBy: { createdAt: 'desc' }, take: 50 });
    return rows.map((l) => this.view(l));
  }

  async create(ctx: ClinicContext, caseId: string, raw: unknown, publicBase: string): Promise<ShareLinkView> {
    const input = parse(createShare, raw);
    const c = await this.prisma.case.findFirst({ where: { id: caseId, clinicId: ctx.clinicId, deletedAt: null }, select: { id: true } });
    if (!c) throw new NotFoundException('Wound not found.');
    const token = randomBytes(32).toString('base64url');
    const link = await this.prisma.shareLink.create({
      data: {
        clinicId: ctx.clinicId,
        caseId,
        tokenHash: this.hash(token),
        hidePersonal: input.hidePersonal,
        createdById: ctx.userId,
        expiresAt: new Date(Date.now() + input.days * DAY_MS),
      },
    });
    await this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'share.create', entity: 'Case', entityId: caseId, details: { days: input.days, hidePersonal: input.hidePersonal } });
    return { ...this.view(link), url: `${publicBase.replace(/\/+$/, '')}/share/${token}` };
  }

  async revoke(ctx: ClinicContext, id: string): Promise<ShareLinkView> {
    const link = await this.prisma.shareLink.findFirst({ where: { id, clinicId: ctx.clinicId } });
    if (!link) throw new NotFoundException('Link not found.');
    const updated = await this.prisma.shareLink.update({ where: { id }, data: { revokedAt: link.revokedAt ?? new Date() } });
    await this.audit.log({ clinicId: ctx.clinicId, userId: ctx.userId, action: 'share.revoke', entity: 'Case', entityId: link.caseId });
    return this.view(updated);
  }

  /** The public page behind a link: the wound's reviewed visits with photos. 410 once expired or revoked. */
  async page(token: string): Promise<{ html: string; imageOrigin: string | null }> {
    if (!/^[\w-]{20,100}$/.test(token)) throw new NotFoundException();
    const link = await this.prisma.shareLink.findUnique({ where: { tokenHash: this.hash(token) } });
    if (!link) throw new NotFoundException();
    if (link.revokedAt || link.expiresAt < new Date()) throw new GoneException('This link is no longer available.');

    const c = await this.prisma.case.findFirst({
      where: { id: link.caseId, deletedAt: null },
      include: { patient: { select: { firstName: true, lastName: true, patientId: true, sex: true } } },
    });
    if (!c) throw new GoneException('This link is no longer available.');
    const results = await this.prisma.aIResult.findMany({
      where: { status: 'ok', phase: { phaseType: 'PRE', treatment: { caseId: c.id, deletedAt: null } } },
      orderBy: { createdAt: 'asc' },
      select: {
        createdAt: true,
        area: true,
        length: true,
        height: true,
        findings: true,
        review: { select: { decision: true, finalReport: true } },
        phase: { select: { treatment: { select: { sequence: true } }, image: { select: { imageUrl: true, thumbPath: true } } } },
      },
    });
    const urls = await this.storage.signedUrls(results.map((r) => r.phase.image?.thumbPath ?? r.phase.image?.imageUrl ?? ''));
    await this.prisma.shareLink.update({ where: { id: link.id }, data: { viewCount: { increment: 1 }, lastViewedAt: new Date() } });
    await this.audit.log({ clinicId: link.clinicId, action: 'share.view', entity: 'Case', entityId: c.id });

    const who = link.hidePersonal ? `Patient ${esc(c.patient.patientId)}` : `${esc(c.patient.firstName)} ${esc(c.patient.lastName)} · ${esc(c.patient.patientId)}`;
    const rows = results
      .map((r) => {
        const type = (r.findings as AnalyzeResponse | null)?.wound_type;
        const review = r.review ? (r.review.decision === 'rejected' ? 'Rejected by clinician' : 'Reviewed') : 'Awaiting review';
        return `<tr><td>T${r.phase.treatment.sequence}</td><td>${fmt(r.createdAt)}</td><td class="n">${r.area ?? '—'}</td><td class="n">${r.length && r.height ? `${r.length} × ${r.height}` : '—'}</td><td>${type ? esc(woundTypeName(type.label)) : '—'}</td><td>${review}</td></tr>`;
      })
      .join('');
    const photos = results
      .map((r) => {
        const path = r.phase.image?.thumbPath ?? r.phase.image?.imageUrl;
        const url = path ? urls.get(path) : null;
        return url ? `<figure><img src="${esc(url)}" alt="T${r.phase.treatment.sequence}"/><figcaption>T${r.phase.treatment.sequence} · ${fmt(r.createdAt)}</figcaption></figure>` : '';
      })
      .join('');
    const latest = [...results].reverse().find((r) => r.review && r.review.decision !== 'rejected');
    const firstUrl = [...urls.values()].find(Boolean);

    const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><meta name="robots" content="noindex"/><title>Wound report</title><style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#111827;margin:32px auto;max-width:860px;padding:0 16px}
.brand{color:#005B4F;font-size:11px;letter-spacing:.8px;text-transform:uppercase;font-weight:700}h1{font-size:22px;margin:6px 0 2px}
.muted{color:#6B7280;font-size:13px}table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
th{text-align:left;color:#6B7280;border-bottom:1px solid #E5E7EB;padding:8px 6px}td{border-bottom:1px solid #F1F3F5;padding:8px 6px}.n{text-align:right}
.photos{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}figure{margin:0;width:160px}img{width:160px;height:160px;object-fit:cover;border-radius:8px;background:#F1F3F5}
figcaption{font-size:11px;color:#6B7280}pre{white-space:pre-wrap;font-family:inherit;font-size:13px;background:#F9FAFB;padding:12px;border-radius:8px}
.note{margin-top:28px;font-size:11px;color:#9CA3AF}</style></head><body>
<div class="brand">Wound report · shared link</div><h1>${who}</h1>
<p class="muted">${esc(c.location)} · ${esc(c.woundType)} · onset ${fmt(c.onset)} · link expires ${fmt(link.expiresAt)}</p>
<table><thead><tr><th>Visit</th><th>Date</th><th class="n">Area cm²</th><th class="n">L × W cm</th><th>Wound type (AI)</th><th>Status</th></tr></thead><tbody>${rows || '<tr><td colspan="6" class="muted">No analysed visits yet</td></tr>'}</tbody></table>
<div class="photos">${photos}</div>
${latest?.review?.finalReport ? `<h2 style="font-size:16px;margin-top:28px">Latest clinician-reviewed report</h2><pre>${esc(latest.review.finalReport)}</pre>` : ''}
<p class="note">Measurements are AI-assisted and reviewed by a clinician. Research prototype, not for patient care. Confidential: share only with people involved in the patient's care.</p>
</body></html>`;
    return { html, imageOrigin: firstUrl ? new URL(firstUrl).origin : null };
  }
}
