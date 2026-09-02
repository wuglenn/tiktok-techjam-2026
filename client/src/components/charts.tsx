"use client";

import { Fragment, type ReactNode } from "react";

import { pp, pct, perturbationLabel, familyOf } from "@/lib/format";
import type { MetricsRow } from "@/lib/types";

export function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
}) {
  return (
    <div className="meta-pair">
      <dt>{label}</dt>
      <dd>
        <div className="tabular">{value}</div>
        {sub != null && <div className="meta-sub">{sub}</div>}
      </dd>
    </div>
  );
}

export function DeltaChip({ v, vs = "clean" }: { v: number; vs?: string }) {
  return (
    <span title={`vs ${vs}`} className="tabular font-mono text-[14px] text-ink-mute">
      {pp(v)}
    </span>
  );
}

export function MetricBar({ value }: { value: number | undefined }) {
  const v = value == null || Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value));
  return (
    <div className="metric-bar">
      <span style={{ width: `${v * 100}%` }} />
    </div>
  );
}

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
    <div className="figure overflow-x-auto">
      <table className="paper-table min-w-[680px]">
        <thead>
          <tr>
            <th>Perturbation</th>
            <th>Macro acc</th>
            <th>Δ vs clean</th>
            <th>F1</th>
            <th>AUROC</th>
            <th>FPR</th>
            <th>FNR</th>
            <th className="text-right">n</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([fam, rows]) => (
            <Fragment key={fam}>
              <tr className="family-row">
                <td colSpan={8}>{fam}</td>
              </tr>
              {rows.map(([key, m]) => {
                const delta =
                  clean?.macro_accuracy != null && m.macro_accuracy != null && key !== "clean"
                    ? m.macro_accuracy - clean.macro_accuracy
                    : null;
                return (
                  <tr key={`${fam}-${key}`}>
                    <td>{perturbationLabel(key)}</td>
                    <td className="tabular text-ink-head">{pct(m.macro_accuracy)}</td>
                    <td>
                      {delta != null ? <DeltaChip v={delta} /> : <span className="text-ink-mute">—</span>}
                    </td>
                    <td className="tabular">{pct(m.f1)}</td>
                    <td className="tabular">{pct(m.auroc)}</td>
                    <td className="tabular">{pct(m.fpr)}</td>
                    <td className="tabular">{pct(m.fnr)}</td>
                    <td className="tabular text-right text-ink-mute">
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
  );
}
