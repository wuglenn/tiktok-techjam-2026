"use client";

import { useEffect, useState } from "react";

import { HeatCanvas, HeatLegend } from "@/components/heat-canvas";
import { IconAlert, IconEye, IconFlask } from "@/components/icons";
import { ProbBar } from "@/components/verdict";
import type { ErrorEntry, EvalResponse } from "@/lib/types";

export default function ErrorsPage() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [active, setActive] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => r.json())
      .then((d: EvalResponse) => setData(d))
      .catch((e) => setError(String(e)));
  }, []);

  const withErrors = (data?.datasets ?? []).filter(
    (d) => d.errors && d.errors.length > 0,
  );
  const ds = withErrors[Math.min(active, Math.max(0, withErrors.length - 1))];
  const fps = ds?.errors?.filter((e) => e.kind === "fp") ?? [];
  const fns = ds?.errors?.filter((e) => e.kind === "fn") ?? [];

  return (
    <div className="space-y-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-white">
            Error analysis
          </h1>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
            Aggregate metrics say <em className="not-italic text-zinc-200">that</em>{" "}
            a detector fails, never <em className="not-italic text-zinc-200">why</em>.
            These are its most confident mistakes, ranked by confidence — a miss at
            P(AI)=0.51 is noise, one at 0.99 is a lesson.
          </p>
        </div>
        <HeatLegend />
      </div>

      {error && (
        <div className="rounded-xl border border-rose-400/20 bg-rose-500/[0.06] px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {data?.mode === "demo" && (
        <div className="rounded-xl border border-amber-400/20 bg-amber-500/[0.06] px-4 py-3 text-xs leading-relaxed text-amber-200/90">
          <strong className="font-semibold">Demo data.</strong>{" "}
          {data.note ??
            "No real eval results found — panels below are illustrative placeholders."}
        </div>
      )}

      {withErrors.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {withErrors.map((d, i) => (
            <button
              key={d.id}
              onClick={() => setActive(i)}
              className={`rounded-full px-4 py-1.5 text-xs font-medium transition-colors ${
                i === active
                  ? "bg-cyan-400 text-zinc-950"
                  : "border border-white/[0.08] bg-white/[0.03] text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {d.name}
            </button>
          ))}
        </div>
      )}

      {fps.length > 0 && (
        <ErrorSection
          kind="fp"
          title="False positives"
          blurb="Real photographs the detector called AI with high confidence — the failure mode that matters most for a tool people trust."
          entries={fps}
        />
      )}
      {fns.length > 0 && (
        <ErrorSection
          kind="fn"
          title="False negatives"
          blurb="Generated images that slipped through as real — usually heavy compression or rescaling destroying the generator's high-frequency fingerprint."
          entries={fns}
        />
      )}

      {!data && !error && (
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="panel p-5">
              <div className="skeleton aspect-video w-full" />
            </div>
          ))}
        </div>
      )}

      {/* trade-offs note */}
      <section className="panel p-6 sm:p-8">
        <div className="flex items-center gap-2.5">
          <IconFlask className="h-4 w-4 text-cyan-300" />
          <h2 className="text-sm font-semibold text-white">
            Trade-offs in this approach
          </h2>
        </div>
        <div className="mt-4 grid gap-4 text-xs leading-relaxed text-zinc-400 md:grid-cols-3">
          <p>
            <strong className="font-semibold text-zinc-200">
              Robustness costs clean accuracy.
            </strong>{" "}
            Wild-simulation augmentation is applied symmetrically to both
            classes, which trades a fraction of a point of clean accuracy for
            holding up under JPEG q30 and heavy blur.
          </p>
          <p>
            <strong className="font-semibold text-zinc-200">
              FPR control is data-bound.
            </strong>{" "}
            False positives concentrate in real images with AI-like statistics —
            digital art, high-ISO noise, heavy bokeh. An eval set without those
            classes cannot reveal them.
          </p>
          <p>
            <strong className="font-semibold text-zinc-200">
              FNs track compression.
            </strong>{" "}
            The missed generators are rarely exotic — they are familiar
            generators behind destructive re-encoding, which is why the mixture
              weights distorted samples so heavily.
          </p>
        </div>
      </section>
    </div>
  );
}

/* ---------------------------------------------------------------- pieces */

function ErrorSection({
  kind,
  title,
  blurb,
  entries,
}: {
  kind: "fp" | "fn";
  title: string;
  blurb: string;
  entries: ErrorEntry[];
}) {
  return (
    <section className="space-y-4">
      <div className="flex items-start gap-3">
        <span
          className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${
            kind === "fp"
              ? "border-rose-400/25 bg-rose-500/10 text-rose-300"
              : "border-sky-400/25 bg-sky-500/10 text-sky-300"
          }`}
        >
          {kind === "fp" ? (
            <IconAlert className="h-4 w-4" />
          ) : (
            <IconEye className="h-4 w-4" />
          )}
        </span>
        <div>
          <h2 className="text-sm font-semibold text-white">{title}</h2>
          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-zinc-500">
            {blurb}
          </p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {entries.map((e) => (
          <ErrorCard key={`${e.kind}-${e.rank}`} e={e} />
        ))}
      </div>
    </section>
  );
}

function ErrorCard({ e }: { e: ErrorEntry }) {
  const src = e.imageAvailable && e.file ? `/api/eval-image?src=${encodeURIComponent(e.file)}` : null;
  return (
    <div className="panel animate-rise overflow-hidden">
      <div className="relative aspect-video">
        <HeatCanvas grid={e.grid ?? null} src={src} opacity={0.6} />
        <div className="absolute left-2.5 top-2.5 flex items-center gap-1.5">
          <span
            className={`rounded-lg px-2 py-1 text-[10px] font-bold uppercase tracking-wider backdrop-blur-sm ${
              e.kind === "fp"
                ? "bg-rose-500/80 text-white"
                : "bg-sky-500/80 text-zinc-950"
            }`}
          >
            {e.kind === "fp" ? "false positive" : "false negative"}
          </span>
          <span className="tabular rounded-lg bg-black/55 px-2 py-1 text-[10px] font-medium text-zinc-300 backdrop-blur-sm">
            #{e.rank}
          </span>
        </div>
        {!src && (
          <span className="absolute bottom-2.5 right-2.5 rounded-lg bg-black/55 px-2 py-1 text-[10px] text-zinc-400 backdrop-blur-sm">
            demo panel — no image on disk
          </span>
        )}
      </div>
      <div className="space-y-3 p-4">
        <div className="flex items-center justify-between gap-3">
          <span className="truncate text-xs font-medium text-zinc-200" title={e.generator}>
            {e.generator ?? "unknown source"}
          </span>
          <span className="shrink-0 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            truth: {e.label === 1 ? "AI" : "real"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="tabular w-12 text-right text-xs font-semibold text-zinc-200">
            {e.prob_ai.toFixed(3)}
          </span>
          <ProbBar p={e.prob_ai} className="flex-1" />
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {(e.distortions ?? []).length > 0 ? (
            (e.distortions ?? []).map((d) => (
              <span
                key={d}
                className="rounded-full border border-white/[0.08] bg-white/[0.03] px-2 py-0.5 text-[10px] text-zinc-400"
              >
                {d}
              </span>
            ))
          ) : (
            <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-2 py-0.5 text-[10px] text-zinc-500">
              clean
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
