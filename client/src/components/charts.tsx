"use client";

import { Fragment, type ReactNode } from "react";

import { pp, pct, perturbationLabel, familyOf } from "@/lib/format";
import type { MetricsRow } from "@/lib/types";

export function StatCard({
  label,
  value,
  sub,
  accent = "cyan",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: "cyan" | "sky" | "emerald" | "rose" | "amber";
}) {
  const accents: Record<string, string> = {
    cyan: "from-cyan-400/60",
    sky: "from-sky-400/60",
    emerald: "from-emerald-400/60",
    rose: "from-rose-400/60",
    amber: "from-amber-400/60",
  };
  return (
    <div className="panel relative overflow-hidden p-5">
      <div
        className={`absolute inset-x-0 top-0 h-px bg-linear-to-r ${accents[accent]} to-transparent`}
      />
      <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-zinc-500">
        {label}
      </p>
      <div className="tabular mt-2 text-2xl font-semibold text-white">{value}</div>
      {sub != null && <div className="mt-1 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

export function DeltaChip({ v, vs = "clean" }: { v: number; vs?: string }) {
  const good = v >= 0;
  return (
    <span
      title={`vs ${vs}`}
      className={`tabular inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium ${
        good
          ? "bg-emerald-500/10 text-emerald-300"
          : "bg-rose-500/10 text-rose-300"
      }`}
    >
      {pp(v)}
    </span>
  );
}

/** Thin inline metric bar, 0..1. */
export function MetricBar({ value }: { value: number | undefined }) {
  const v = value == null || Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value));
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.05]">
      <div
        className="h-full rounded-full bg-cyan-400/60 transition-[width] duration-700"
        style={{ width: `${v * 100}%` }}
      />
    </div>
  );
}

/** Grouped metric table rows for one sweep. */
export function SweepTable({ sweep }: { sweep: Record<string, MetricsRow> }) {
  const families = new Map<string, [string, MetricsRow][]>();
  for (const [key, m] of Object.entries(sweep)) {
    const fam = key === "clean" ? "Clean" : familyOf(key);
    if (!families.has(fam)) families.set(fam, []);
    families.get(fam)!.push([key, m]);
  }
  const order = ["Clean", "Compression", "Blur", "Rescale", "Noise", "Color", "Geometry", "Protocol", "Other"];
  const rank = (f: string) => {
    const i = order.indexOf(f);
    return i === -1 ? 99 : i;
  };
  const sorted = [...families.entries()].sort((a, b) => rank(a[0]) - rank(b[0]));

  const clean = sweep["clean"];

  return (
    <div className="panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[620px] text-sm">
          <thead>
            <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
              <th className="px-5 py-3.5 font-medium">Perturbation</th>
              <th className="px-4 py-3.5 font-medium">Macro acc</th>
              <th className="px-4 py-3.5 font-medium">Δ vs clean</th>
              <th className="px-4 py-3.5 font-medium">F1</th>
              <th className="px-4 py-3.5 font-medium">FPR</th>
              <th className="px-4 py-3.5 font-medium">FNR</th>
              <th className="px-5 py-3.5 text-right font-medium">n</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(([fam, rows]) => (
              <Fragment key={fam}>
                <tr className="bg-white/[0.02]">
                  <td
                    colSpan={7}
                    className="px-5 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500"
                  >
                    {fam}
                  </td>
                </tr>
                {rows.map(([key, m]) => {
                  const delta =
                    clean?.macro_accuracy != null && m.macro_accuracy != null && key !== "clean"
                      ? m.macro_accuracy - clean.macro_accuracy
                      : null;
                  return (
                    <tr
                      key={`${fam}-${key}`}
                      className={`border-b border-white/[0.04] transition-colors hover:bg-white/[0.03] ${
                        key === "clean" ? "bg-cyan-500/[0.04]" : ""
                      }`}
                    >
                      <td className="px-5 py-3 text-zinc-200">{perturbationLabel(key)}</td>
                      <td className="tabular px-4 py-3 font-medium text-white">
                        <div className="flex items-center gap-2.5">
                          <span className="w-12">{pct(m.macro_accuracy)}</span>
                          <div className="w-20">
                            <MetricBar value={m.macro_accuracy} />
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        {delta != null ? <DeltaChip v={delta} /> : <span className="text-zinc-600">—</span>}
                      </td>
                      <td className="tabular px-4 py-3 text-zinc-300">{pct(m.f1)}</td>
                      <td className="tabular px-4 py-3 text-rose-300/90">{pct(m.fpr)}</td>
                      <td className="tabular px-4 py-3 text-amber-300/90">{pct(m.fnr)}</td>
                      <td className="tabular px-5 py-3 text-right text-zinc-500">
                        {m.n?.toLocaleString() ?? "–"}
                      </td>
                    </tr>
                  );
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
