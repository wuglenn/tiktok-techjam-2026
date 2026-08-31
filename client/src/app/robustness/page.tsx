"use client";

import { useEffect, useMemo, useState } from "react";

import { DeltaChip, StatCard, SweepTable } from "@/components/charts";
import { IconChart, IconShield } from "@/components/icons";
import { pct, perturbationLabel } from "@/lib/format";
import type { EvalDataset, EvalResponse, MetricsRow } from "@/lib/types";

export default function RobustnessPage() {
  const [data, setData] = useState<EvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => r.json())
      .then((d: EvalResponse) => setData(d))
      .catch((e) => setError(String(e)));
  }, []);

  const datasets = (data?.datasets ?? []).filter(
    (d) => d.sweep && Object.keys(d.sweep).length > 1,
  );
  const ds: EvalDataset | undefined = datasets[Math.min(active, Math.max(0, datasets.length - 1))];

  const stats = useMemo(() => {
    if (!ds?.sweep) return null;
    const clean = ds.sweep["clean"]?.macro_accuracy ?? null;
    const rows = Object.entries(ds.sweep).filter(([k]) => k !== "clean");
    const worst = rows.reduce<[string, MetricsRow] | null>(
      (acc, [k, m]) =>
        !acc || (m.macro_accuracy ?? 1) < (acc[1].macro_accuracy ?? 1) ? [k, m] : acc,
      null,
    );
    const meanDelta =
      clean != null && rows.length
        ? rows.reduce((s, [, m]) => s + ((m.macro_accuracy ?? clean) - clean), 0) / rows.length
        : null;
    const worstFpr = rows.reduce((mx, [, m]) => Math.max(mx, m.fpr ?? 0), 0);
    return { clean, worst, meanDelta, worstFpr };
  }, [ds]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-white">
          Robustness
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-zinc-400">
          Clean versus transformed performance — the benchmark perturbation
          protocol (JPEG, blur, resize, noise, jitter, crop) plus the Pangram
          augmented protocol, applied symmetrically to both classes.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-400/20 bg-rose-500/[0.06] px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {data?.mode === "demo" && <DemoNote note={data.note} />}

      {!data && !error && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="panel p-5">
              <div className="skeleton h-3 w-20" />
              <div className="skeleton mt-3 h-7 w-24" />
            </div>
          ))}
        </div>
      )}

      {datasets.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {datasets.map((d, i) => (
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

      {ds?.file && (
        <p className="text-xs text-zinc-600">
          source: <span className="tabular text-zinc-500">{ds.file}</span>
        </p>
      )}

      {stats && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Clean macro acc"
            value={stats.clean != null ? `${pct(stats.clean)}%` : "–"}
            sub={ds ? `${ds.metrics.n?.toLocaleString() ?? "–"} images` : undefined}
            accent="cyan"
          />
          <StatCard
            label="Worst perturbation"
            value={stats.worst ? perturbationLabel(stats.worst[0]) : "–"}
            sub={
              stats.worst ? (
                <span className="flex items-center gap-1.5">
                  {pct(stats.worst[1].macro_accuracy)}%
                  {stats.clean != null && (
                    <DeltaChip v={(stats.worst[1].macro_accuracy ?? 0) - stats.clean} />
                  )}
                </span>
              ) : undefined
            }
            accent="rose"
          />
          <StatCard
            label="Mean degradation"
            value={stats.meanDelta != null ? `${(stats.meanDelta * 100).toFixed(2)} pp` : "–"}
            sub="macro accuracy across all perturbations"
            accent="amber"
          />
          <StatCard
            label="Worst FPR"
            value={`${pct(stats.worstFpr)}%`}
            sub="real images called AI, worst pass"
            accent="sky"
          />
        </div>
      )}

      {ds?.sweep && <SweepTable sweep={ds.sweep} />}

      {ds?.per_distorted && <DistortedPanel ds={ds} />}
    </div>
  );
}

/* ---------------------------------------------------------------- pieces */

function DemoNote({ note }: { note?: string }) {
  return (
    <div className="rounded-xl border border-amber-400/20 bg-amber-500/[0.06] px-4 py-3 text-xs leading-relaxed text-amber-200/90">
      <strong className="font-semibold">Demo data.</strong>{" "}
      {note ?? "No real eval results found — numbers shown are placeholders."}
    </div>
  );
}

/** NTIRE-style clean vs distorted split (labelled distortions in the eval set). */
function DistortedPanel({ ds }: { ds: EvalDataset }) {
  const clean = ds.per_distorted?.["clean"];
  const distorted = ds.per_distorted?.["distorted"];
  if (!clean || !distorted) return null;
  return (
    <section className="space-y-4">
      <div className="flex items-center gap-2.5">
        <IconShield className="h-4 w-4 text-cyan-400" />
        <h2 className="text-sm font-semibold text-white">
          Clean vs distorted labels (NTIRE protocol)
        </h2>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="panel p-5">
          <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-zinc-500">
            Clean images
          </p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="tabular text-2xl font-semibold text-white">
              {pct(clean.macro_accuracy)}%
            </span>
            <span className="text-xs text-zinc-500">macro accuracy</span>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            AUROC {pct(clean.auroc)}% · n={clean.n?.toLocaleString() ?? "–"}
          </p>
        </div>
        <div className="panel p-5">
          <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-zinc-500">
            Distorted images
          </p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="tabular text-2xl font-semibold text-white">
              {pct(distorted.macro_accuracy)}%
            </span>
            {clean.macro_accuracy != null && distorted.macro_accuracy != null && (
              <DeltaChip v={distorted.macro_accuracy - clean.macro_accuracy} />
            )}
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            AUROC {pct(distorted.auroc)}% · n={distorted.n?.toLocaleString() ?? "–"}
            {ds.robust_n != null && ` · robust AUROC ${pct(ds.robust_auroc)}%`}
          </p>
        </div>
      </div>

      {ds.per_distortion && Object.keys(ds.per_distortion).length > 0 && (
        <div className="panel overflow-hidden">
          <div className="border-b border-white/[0.06] px-5 py-3.5">
            <div className="flex items-center gap-2">
              <IconChart className="h-4 w-4 text-zinc-500" />
              <span className="text-xs font-semibold text-white">
                By first distortion type
              </span>
            </div>
          </div>
          <div className="divide-y divide-white/[0.04]">
            {Object.entries(ds.per_distortion)
              .sort((a, b) => (b[1].macro_accuracy ?? 0) - (a[1].macro_accuracy ?? 0))
              .map(([k, m]) => (
                <div key={k} className="grid grid-cols-[1fr_auto] items-center gap-4 px-5 py-3">
                  <span className="text-sm capitalize text-zinc-300">{k}</span>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="tabular text-zinc-500">
                      n={m.n?.toLocaleString() ?? "–"}
                    </span>
                    <span className="tabular w-14 text-right font-medium text-white">
                      {pct(m.macro_accuracy)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </section>
  );
}
