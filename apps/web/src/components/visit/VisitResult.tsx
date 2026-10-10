"use client"
import { useState, type ReactNode } from "react";
import { AlertOctagon, AlertTriangle, Ban, CheckCircle2, ClipboardCheck, Download, PencilLine, Trash2, XCircle } from "lucide-react";
import {
  isUncertain,
  woundTypeName,
  type Care,
  type ClassResult,
  type Outline,
  type AnalyzeResponse,
  type PhotoChange,
  type Progress,
  type ReviewDecision,
  type SuggestionDecisions,
} from "@antigravity-project-spec-pack/domain/wound-model";
import type { ReviewView, VisitView } from "@antigravity-project-spec-pack/domain/api";
import { errorMessage } from "../../lib/api";
import { useDeleteVisit, useReviewVisit } from "../../lib/queries";
import { dateTimeText } from "../../lib/format";
import { Button } from "../ui/button";
import { fieldClass } from "../ui/field";

const pct = (p: number) => `${Math.round(p * 100)}%`;

/** Tissue layer colours: the colours clinicians already associate with each tissue. */
export const TISSUE_COLOR: Record<string, string> = {
  granulation: "#e11d48",
  slough: "#eab308",
  necrosis: "#111827",
  epithelial: "#f9a8d4",
  exposed_structure: "#f8fafc",
  periwound_erythema: "#f97316",
  maceration: "#93c5fd",
  callus: "#a3e635",
};

/** The photo with the wound outline (and optionally the tissue layers) drawn over it. Points are 0–1, so the
 * polygons fit any display size. */
export function WoundOverlay({ src, outline, tissue }: { src: string; outline?: Outline | null; tissue?: Record<string, Outline> | null }) {
  return (
    <div className="relative w-full overflow-hidden rounded-2xl bg-black/5">
      <img src={src} alt="Wound photo with the outline found by the model" className="block w-full" />
      {tissue && (
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden>
          {Object.entries(tissue).flatMap(([name, polygons]) =>
            polygons.map((polygon, i) => (
              <polygon
                key={`${name}-${i}`}
                points={polygon.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")}
                fill={TISSUE_COLOR[name] ?? "#a855f7"}
                fillOpacity={0.45}
                stroke={TISSUE_COLOR[name] ?? "#a855f7"}
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
            )),
          )}
        </svg>
      )}
      {outline && outline.length > 0 && (
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden>
          {outline.map((polygon, i) => (
            <polygon
              key={i}
              points={polygon.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")}
              fill="rgba(57, 255, 20, 0.12)"
              stroke="#39ff14"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      )}
    </div>
  );
}

/** A classification as the report writes it: a confident label, or UNCERTAIN with the top estimates. */
function classText(result: ClassResult | undefined, name: (label: string) => string): string {
  if (!result) return "Not assessed yet (model not installed)";
  if (result.rule) {
    return `${name(result.label)} (by rule: ${result.rule}${result.model ? `; model estimate: ${classText(result.model, name)}` : ""})`;
  }
  if (result.prob === null) return name(result.label);
  if (isUncertain(result)) {
    return `Uncertain — top estimates: ${result.top.map(([label, p]) => `${name(label)} ${pct(p)}`).join(", ")}`;
  }
  return `${name(result.label)} (model confidence ${pct(result.prob)})`;
}

/** The draft's markdown (headings, bullets, **bold**, _italics_) as plain elements; never injected as HTML. */
function ReportText({ markdown }: { markdown: string }) {
  const inline = (text: string): ReactNode[] =>
    text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
      part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
    );
  return (
    <div className="space-y-1.5 text-sm leading-relaxed">
      {markdown.split("\n").map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-1" />;
        if (line.startsWith("## ")) return <h5 key={i} className="pt-2 font-semibold">{line.slice(3)}</h5>;
        if (line.startsWith("# ")) return <h4 key={i} className="text-base font-semibold">{line.slice(2)}</h4>;
        if (line.startsWith("- ")) return <p key={i} className="pl-4 -indent-3">• {inline(line.slice(2))}</p>;
        if (/^_.*_$/.test(line)) return <p key={i} className="italic text-muted-foreground">{line.slice(1, -1)}</p>;
        return <p key={i}>{inline(line)}</p>;
      })}
    </div>
  );
}

const DECISION: Record<ReviewDecision, { label: string; icon: typeof CheckCircle2; tone: string }> = {
  approved: { label: "Approved", icon: CheckCircle2, tone: "text-emerald-700 bg-emerald-50 border-emerald-200" },
  edited: { label: "Edited and approved", icon: PencilLine, tone: "text-sky-800 bg-sky-50 border-sky-200" },
  rejected: { label: "Rejected", icon: XCircle, tone: "text-red-700 bg-red-50 border-red-200" },
};

export function ReviewBadge({ review }: { review: ReviewView | null }) {
  if (!review) {
    return <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800">Awaiting review</span>;
  }
  const { label, icon: Icon, tone } = DECISION[review.decision];
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${tone}`}>
      <Icon className="w-3.5 h-3.5" /> {label}
    </span>
  );
}

/** Approve, edit or reject the draft. One decision per result; it is stored with the reviewer and time. */
function ReviewPanel({ caseId, visit }: { caseId: string; visit: VisitView }) {
  const [mode, setMode] = useState<"choose" | "edit" | "reject">("choose");
  const [text, setText] = useState(visit.draftReport ?? "");
  const [reason, setReason] = useState("");
  const [problem, setProblem] = useState<string>();
  const [decisions, setDecisions] = useState<SuggestionDecisions>({});
  const suggestions = visit.care?.suggestions ?? [];
  const review = useReviewVisit(caseId, visit.id);
  const busy = review.isPending;

  const submit = (decision: ReviewDecision) => {
    if (decision === "edited" && !text.trim()) return setProblem("Write the corrected report.");
    if (decision === "rejected" && !reason.trim()) return setProblem("Say why the draft is rejected.");
    if (decision !== "rejected" && suggestions.some((s) => !decisions[s.rule_id])) {
      return setProblem("Accept or decline each care suggestion.");
    }
    setProblem(undefined);
    review.mutate(
      {
        decision,
        finalReport: decision === "edited" ? text : undefined,
        reason: decision === "rejected" ? reason : undefined,
        // Which rule-based suggestions the clinician agreed with: the record the rules are tuned against.
        corrections: suggestions.length && visit.rulesVersion ? { suggestions: decisions, rulesVersion: visit.rulesVersion } : undefined,
      },
      { onError: (e) => setProblem(errorMessage(e)) },
    );
  };

  return (
    <div className="rounded-2xl border border-black/10 p-4 dark:border-white/10">
      <p className="text-sm font-semibold">Clinician review</p>
      <p className="text-xs text-muted-foreground mt-0.5">Nothing in this draft is part of the record until you approve or edit it.</p>
      {visit.care && mode !== "reject" && <CareList care={visit.care} decisions={decisions} onDecide={(id, d) => setDecisions((x) => ({ ...x, [id]: d }))} />}
      {mode === "edit" && (
        <textarea aria-label="Corrected report" className={`${fieldClass} mt-3 min-h-64 font-mono text-xs`} value={text} onChange={(e) => setText(e.target.value)} />
      )}
      {mode === "reject" && (
        <textarea
          aria-label="Reason for rejecting"
          placeholder="For example: wrong wound outlined, photo of the wrong site"
          className={`${fieldClass} mt-3 min-h-24`}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />
      )}
      {problem && (
        <p role="alert" className="mt-2 text-sm font-medium text-destructive">
          {problem}
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {mode === "choose" ? (
          <>
            <Button type="button" size="sm" disabled={busy} onClick={() => submit("approved")}>
              {busy ? "Saving…" : "Approve draft"}
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={() => setMode("edit")}>
              Edit
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={() => setMode("reject")}>
              Reject
            </Button>
          </>
        ) : (
          <>
            <Button type="button" size="sm" disabled={busy} onClick={() => submit(mode === "edit" ? "edited" : "rejected")}>
              {busy ? "Saving…" : mode === "edit" ? "Save and approve" : "Reject draft"}
            </Button>
            <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => (setMode("choose"), setProblem(undefined))}>
              Cancel
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * The full-size photo with the outline, and the tissue layers if given, drawn on it as a JPG: the same drawing as
 * WoundOverlay, at the photo's own resolution. Points are 0–1 of the photo, as the model returns them.
 */
async function outlinedJpeg(src: string, outline: Outline | null | undefined, tissue: Record<string, Outline> | null | undefined): Promise<Blob> {
  const res = await fetch(src);
  if (!res.ok) throw new Error("The photo couldn't be downloaded. Reload the page and try again.");
  const photo = await createImageBitmap(await res.blob());
  const canvas = document.createElement("canvas");
  canvas.width = photo.width;
  canvas.height = photo.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("This browser can't draw the image.");
  ctx.drawImage(photo, 0, 0);
  const trace = (polygon: [number, number][]) => {
    ctx.beginPath();
    polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x * photo.width, y * photo.height) : ctx.moveTo(x * photo.width, y * photo.height)));
    ctx.closePath();
  };
  for (const [name, polygons] of Object.entries(tissue ?? {})) {
    ctx.fillStyle = ctx.strokeStyle = TISSUE_COLOR[name] ?? "#a855f7";
    ctx.lineWidth = Math.max(1, Math.round(photo.width / 600));
    for (const polygon of polygons) {
      trace(polygon);
      ctx.globalAlpha = 0.45;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
  }
  ctx.fillStyle = "rgba(57, 255, 20, 0.12)";
  ctx.strokeStyle = "#39ff14";
  ctx.lineWidth = Math.max(2, Math.round(photo.width / 300)); // as the model service's own overlay
  for (const polygon of outline ?? []) {
    trace(polygon);
    ctx.fill();
    ctx.stroke();
  }
  photo.close();
  const jpg = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  if (!jpg) throw new Error("The image couldn't be saved.");
  return jpg;
}

function DownloadOutlined({ src, outline, tissue, filename }: { src: string; outline: Outline; tissue?: Record<string, Outline> | null; filename: string }) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string>();
  const download = async () => {
    setBusy(true);
    setProblem(undefined);
    try {
      const url = URL.createObjectURL(await outlinedJpeg(src, outline, tissue));
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (e) {
      setProblem(e instanceof Error ? e.message : "The image couldn't be saved.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        disabled={busy}
        onClick={download}
        className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline disabled:opacity-50"
      >
        <Download className="w-4 h-4" /> {busy ? "Preparing…" : tissue ? "Download with outline and tissue" : "Download with outline"}
      </button>
      {problem && (
        <span role="alert" className="text-sm text-destructive">
          {problem}
        </span>
      )}
    </div>
  );
}

function PhasePhoto({
  label,
  src,
  outline,
  tissue,
  note,
  filename,
}: {
  label?: string;
  src: string | null;
  outline?: Outline | null;
  tissue?: Record<string, Outline> | null;
  note?: string;
  /** Name for the downloaded outlined photo: no patient details, which would travel with the file. */
  filename: string;
}) {
  return (
    <figure className="flex flex-col gap-1.5">
      {src ? (
        <WoundOverlay src={src} outline={outline} tissue={tissue} />
      ) : (
        <div className="flex min-h-48 items-center justify-center rounded-2xl bg-black/5 text-sm text-muted-foreground">Photo unavailable</div>
      )}
      {label && <figcaption className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</figcaption>}
      {note && <p className="text-xs text-muted-foreground">{note}</p>}
      {src && outline && outline.length > 0 && <DownloadOutlined src={src} outline={outline} tissue={tissue} filename={filename} />}
    </figure>
  );
}

const TRAJECTORY: Record<string, { label: string; tone: string }> = {
  improving: { label: "Improving", tone: "text-emerald-800 bg-emerald-50 border-emerald-200" },
  static: { label: "Static", tone: "text-amber-800 bg-amber-50 border-amber-200" },
  deteriorating: { label: "Deteriorating", tone: "text-red-800 bg-red-50 border-red-200" },
};

const changeText = (c: PhotoChange | undefined): string | null => {
  if (!c) return null;
  const parts: string[] = [];
  if (c.percent_area_reduction !== undefined) {
    parts.push(`${c.area_before_cm2} → ${c.area_after_cm2} cm² (${Math.abs(c.percent_area_reduction)}% ${c.percent_area_reduction >= 0 ? "smaller" : "larger"})`);
    if (c.cm2_per_week !== undefined) parts.push(`${c.cm2_per_week} cm²/week`);
  }
  if (c.nonviable_before_pct !== undefined) parts.push(`non-viable tissue ${c.nonviable_before_pct}% → ${c.nonviable_after_pct}%`);
  if (!parts.length) return null;
  return `${parts.join(" · ")}${c.days ? ` over ${c.days} days` : ""}`;
};

/** How the wound is healing visit to visit (like with like), and what this visit's cleaning did. */
function HealingPanel({ progress }: { progress: Progress }) {
  const h = progress.healing;
  const s = progress.session;
  const t = h.trajectory ? TRAJECTORY[h.trajectory] : undefined;
  const rows: [string, string | null][] = [
    ["Since last visit", changeText(h.since_last)],
    ["Since first visit", changeText(h.since_first)],
    [
      "4-week check",
      h.four_week
        ? `${h.four_week.percent_area_reduction}% smaller at day ${h.four_week.days} (target ${h.four_week.target}%): ${h.four_week.on_track ? "on track" : "not on track"}`
        : null,
    ],
    [
      "This visit",
      s
        ? [
            s.area_before_cm2 !== undefined ? `${s.area_before_cm2} → ${s.area_after_cm2} cm²${s.area_note ? ` (${s.area_note})` : ""}` : null,
            s.nonviable_before_pct !== undefined ? `non-viable tissue ${s.nonviable_before_pct}% → ${s.nonviable_after_pct}%` : null,
          ]
            .filter(Boolean)
            .join(" · ") || null
        : null,
    ],
    ["PUSH score", progress.push ? `${progress.push.score}/17 (${progress.push.note})` : null],
  ];
  return (
    <section className="rounded-2xl border border-black/10 p-4 dark:border-white/10">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-sm font-semibold">Healing</p>
        {t ? (
          <span className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${t.tone}`}>{t.label}</span>
        ) : (
          <span className="text-xs text-muted-foreground">Not compared: {h.reason ?? "not enough photos"}</span>
        )}
        {h.trajectory && (
          <span className="text-xs text-muted-foreground">
            by {h.basis}, comparing {h.phase === "post" ? "after-cleaning" : "before-treatment"} photos; changes within measurement noise
            {h.noise_band ? ` (${h.noise_band.smaller_pct}% smaller to ${h.noise_band.larger_pct}% larger; ${h.noise_band.source})` : ""} count as static
          </span>
        )}
      </div>
      <dl className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
        {rows
          .filter((r): r is [string, string] => r[1] !== null)
          .map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</dt>
              <dd className="mt-0.5">{value}</dd>
            </div>
          ))}
      </dl>
    </section>
  );
}

/** The rule-based care suggestions: checks and things to avoid first, then each suggestion with its reason. */
function CareList({ care, decisions, onDecide }: { care: Care; decisions: SuggestionDecisions; onDecide: (ruleId: string, d: "accepted" | "declined") => void }) {
  if (!care.checks.length && !care.contraindications.length && !care.suggestions.length) return null;
  return (
    <div className="mt-3 flex flex-col gap-2 text-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Care suggestions (placeholder rules, not yet clinically signed off)</p>
      {care.checks.map((c) => (
        <p key={c.rule_id} className="flex gap-2">
          <ClipboardCheck className="w-4 h-4 mt-0.5 shrink-0 text-sky-700" /> {c.text}
        </p>
      ))}
      {care.contraindications.map((c) => (
        <p key={c.action} className="flex gap-2 text-red-800">
          <Ban className="w-4 h-4 mt-0.5 shrink-0" />
          <span>
            <strong className="font-semibold">Avoid {c.action}:</strong> {c.reason}.
          </span>
        </p>
      ))}
      <ul className="flex flex-col gap-2">
        {care.suggestions.map((s) => {
          const d = decisions[s.rule_id];
          return (
            <li key={s.rule_id} className="rounded-xl border border-black/10 p-3 dark:border-white/10">
              <p>
                <strong className="font-semibold">{s.action}</strong>
                {s.alternatives.length > 0 && <span className="text-muted-foreground"> (or {s.alternatives.join(", ")})</span>}: {s.text}
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">Because: {s.because.join("; ")}</p>
              <div className="mt-2 flex gap-2" role="group" aria-label={`Decision on ${s.action}`}>
                <Button type="button" size="sm" variant={d === "accepted" ? "default" : "outline"} aria-pressed={d === "accepted"} onClick={() => onDecide(s.rule_id, "accepted")}>
                  Accept
                </Button>
                <Button type="button" size="sm" variant={d === "declined" ? "default" : "outline"} aria-pressed={d === "declined"} onClick={() => onDecide(s.rule_id, "declined")}>
                  Decline
                </Button>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Deletes the visit and its photo after a confirmation; there is no undo. */
export function DeleteVisit({ caseId, visit }: { caseId: string; visit: VisitView }) {
  const del = useDeleteVisit(caseId);
  const busy = del.isPending;
  const problem = del.error ? errorMessage(del.error) : undefined;
  const remove = () => {
    if (!window.confirm("Delete this visit, its photo and its review? This can't be undone.")) return;
    del.mutate(visit.id);
  };
  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-black/5 pt-3 dark:border-white/10">
      <button type="button" disabled={busy} onClick={remove} className="inline-flex items-center gap-1.5 text-sm font-medium text-destructive hover:underline disabled:opacity-50">
        <Trash2 className="w-4 h-4" /> {busy ? "Deleting…" : "Delete visit"}
      </button>
      {problem && (
        <span role="alert" className="text-sm text-destructive">
          {problem}
        </span>
      )}
    </div>
  );
}

/** One analysed photo: flags first, the outlined photo, the findings, the draft, and the review. */
export function VisitResult({ caseId, visit, canDelete = true }: { caseId: string; visit: VisitView; canDelete?: boolean }) {
  const f: AnalyzeResponse = visit.findings ?? { status: "ok" }; // shown only once the analysis is done
  const [showTissue, setShowTissue] = useState(false);
  const hasTissue = !!(f.tissue_outline || visit.post?.findings?.tissue_outline);
  const m = f.measurement;
  const change = f.change?.percent_area_reduction;
  const flags = [...(f.flags ?? []), ...(visit.progress?.flags ?? []).filter((p) => !f.flags?.some((x) => x.text === p.text))].sort((a, b) =>
    a.level === b.level ? 0 : a.level === "urgent" ? -1 : 1,
  );

  const findings: [string, string][] = [
    ["Wound type", classText(f.wound_type, woundTypeName)],
    ...Object.entries(f.severity ?? {}).map(([head, result]): [string, string] => [
      head.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()),
      classText(result, (label) => label.replace(/_/g, " ")),
    ]),
    [
      "Size",
      m
        ? `${m.area_cm2} cm² · ${m.length_cm} × ${m.width_cm} cm · perimeter ${m.perimeter_cm} cm`
        : f.marker_found === false
          ? "Not measured: calibration sticker not found in the photo"
          : "Not measured: no wound outline yet (outline model not installed)",
    ],
    ...(change !== undefined && !visit.progress
      ? [[
          "Change",
          `${change >= 0 ? "Smaller" : "Larger"} by ${Math.abs(change)}% since the last measured photo${f.change?.days_between ? ` (${f.change.days_between} days)` : ""}`,
        ] as [string, string]]
      : []),
    ...(f.tissue_pct ? [["Tissue", Object.entries(f.tissue_pct).map(([k, v]) => `${k} ${v}%`).join(", ")] as [string, string]] : []),
    ...(f.tissue_untrusted?.length
      ? [["Possibly also", `${f.tissue_untrusted.map((c) => c.replace(/_/g, " ")).join(", ")} (the tissue model is not yet reliable for these: check on examination)`] as [string, string]]
      : []),
    ...(!f.tissue_pct && f.tissue_pct_uncertain
      ? [[
          "Tissue",
          `Uncertain (model confidence ${pct(f.tissue_confidence ?? 0)}): ${Object.entries(f.tissue_pct_uncertain).map(([k, v]) => `${k} ${v}%`).join(", ")}. Assess on examination.`,
        ] as [string, string]]
      : []),
  ];

  return (
    <div className="flex flex-col gap-5">
      <p className="rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white">
        AI draft for clinician review, not a diagnosis.
      </p>

      {flags.length > 0 && (
        <ul className="flex flex-col gap-2">
          {flags.map((flag) => {
            const urgent = flag.level === "urgent";
            const Icon = urgent ? AlertOctagon : AlertTriangle;
            return (
              <li
                key={flag.text}
                className={`flex gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm ${
                  urgent ? "border-red-300 bg-red-50 text-red-900" : "border-amber-300 bg-amber-50 text-amber-900"
                }`}
              >
                <Icon className="w-4 h-4 mt-0.5 shrink-0" />
                <span>
                  <strong className="font-semibold">{urgent ? "Urgent: " : "Review: "}</strong>
                  {flag.text}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        <div className={visit.post ? "grid grid-cols-2 gap-3 self-start" : ""}>
          <PhasePhoto
            label={visit.post ? "Before treatment" : undefined}
            src={visit.photoUrl}
            outline={f.outline}
            tissue={showTissue ? f.tissue_outline : null}
            filename={`wound-T${visit.sequence}-before-outlined.jpg`}
          />
          {visit.post && (
            <PhasePhoto
              label="After cleaning"
              src={visit.post.photoUrl}
              outline={visit.post.findings?.outline}
              tissue={showTissue ? visit.post.findings?.tissue_outline : null}
              filename={`wound-T${visit.sequence}-after-cleaning-outlined.jpg`}
              note={
                visit.post.status === "processing"
                  ? "Analysing…"
                  : visit.post.status === "ok"
                    ? undefined
                    : "Kept for the record; the model could not analyse it"
              }
            />
          )}
        </div>
        <dl className="flex flex-col gap-3 text-sm">
          {findings.map(([label, value]) => (
            <div key={label}>
              <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</dt>
              <dd className="mt-0.5">{value}</dd>
            </div>
          ))}
          <div>
            <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Analysed</dt>
            <dd className="mt-0.5">{dateTimeText(visit.takenAt)}</dd>
          </div>
          {hasTissue && (
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showTissue} onChange={(e) => setShowTissue(e.target.checked)} />
              Show tissue layers
              <span className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {Object.entries(TISSUE_COLOR).map(([name, color]) => (
                  <span key={name} className="inline-flex items-center gap-1">
                    <span className="inline-block h-2.5 w-2.5 rounded-sm border border-black/20" style={{ background: color }} />
                    {name.replace(/_/g, " ")}
                  </span>
                ))}
              </span>
            </label>
          )}
        </dl>
      </div>

      {visit.progress && <HealingPanel progress={visit.progress} />}

      {visit.review ? (
        <div className="rounded-2xl border border-black/10 p-4 dark:border-white/10">
          <div className="flex flex-wrap items-center gap-2">
            <ReviewBadge review={visit.review} />
            <span className="text-xs text-muted-foreground">{dateTimeText(visit.review.createdAt)}</span>
          </div>
          {visit.review.reason && <p className="mt-2 text-sm">Reason: {visit.review.reason}</p>}
          {visit.review.finalReport && (
            <div className="mt-3">
              <ReportText markdown={visit.review.finalReport} />
            </div>
          )}
        </div>
      ) : (
        <>
          {visit.draftReport && (
            <details className="rounded-2xl border border-black/10 p-4 dark:border-white/10" open>
              <summary className="cursor-pointer text-sm font-semibold">Draft report</summary>
              <div className="mt-3">
                <ReportText markdown={visit.draftReport} />
              </div>
            </details>
          )}
          <ReviewPanel caseId={caseId} visit={visit} />
        </>
      )}
      {canDelete && <DeleteVisit caseId={caseId} visit={visit} />}
    </div>
  );
}
