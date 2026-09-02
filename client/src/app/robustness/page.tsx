"use client";

import { useEffect, useMemo, useState } from "react";

import { DeltaChip, StatCard, SweepTable } from "@/components/charts";
import { Measure, Notice, Tabs } from "@/components/essay";
import { NtireLeaderboard } from "@/components/ntire-leaderboard";
import { evalDisplayName } from "@/lib/eval-labels";
import { pct, perturbationLabel } from "@/lib/format";
import type { EvalDataset, EvalResponse, MetricsRow } from "@/lib/types";

function isRobust(d: EvalDataset): boolean {
  if (d.sweep && Object.keys(d.sweep).length > 1) return true;
  if (d.per_distorted && Object.keys(d.per_distorted).length > 0) return true;
  if (d.per_distortion && Object.keys(d.per_distortion).length > 0) return true;
  return false;
}

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

  const datasets = (data?.datasets ?? []).filter(isRobust);
  const ds: EvalDataset | undefined = datasets[Math.min(active, Math.max(0, datasets.length - 1))];

  const stats = useMemo(() => {
    if (!ds) return null;
    const clean =
      ds.per_distorted?.clean?.macro_accuracy ??
      ds.sweep?.clean?.macro_accuracy ??
      null;
    const rows: [string, MetricsRow][] = ds.sweep
      ? Object.entries(ds.sweep).filter(([k]) => k !== "clean")
      : Object.entries(ds.per_distortion ?? {}).filter(
          ([k]) => k !== "clean" && k !== "none",
        );
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
    <div className="space-y-10">
      <Measure className="essay">
        <h1 className="essay-title">Robustness under common transformations</h1>
        <p className="mt-4">
          We evaluate Seer on 13,843 held-out images from the TechJam evaluation
          set: 5,000 real COCO photographs and 8,843 WildFake DALL·E 3 images.
          The sweep measures clean performance and 15 common transformations,
          including recompression, blur, resizing, noise, color jitter, cropping,
          and the pangram submission protocol.
        </p>
        <p className="mt-4">
          For comparison, the table below is the NTIRE 2026 public-test
          leaderboard from Table 3 of{" "}
          <a
            href="https://arxiv.org/pdf/2604.11487"
            target="_blank"
            rel="noreferrer"
          >
            Gushchin et al.
          </a>
          . The published entries are 7 billion parameter models. Our model,
          Seer, at 302 million parameters sits third on robust ROC AUC, state
          of the art at this scale.
        </p>
      </Measure>

      <NtireLeaderboard datasets={data?.datasets} />

      {error && (
        <Measure>
          <Notice>{error}</Notice>
        </Measure>
      )}

      {data?.mode === "demo" && (
        <Measure>
          <Notice>
            Demo data. {data.note ?? "No real eval results found — numbers shown are placeholders."}
          </Notice>
        </Measure>
      )}

      {!data && !error && (
        <Measure>
          <div className="meta-grid">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="meta-pair">
                <div className="skeleton h-4 w-20" />
                <div className="skeleton h-4 w-16 justify-self-end" />
              </div>
            ))}
          </div>
        </Measure>
      )}

      {datasets.length > 1 && (
        <Measure>
          <Tabs
            items={datasets.map((d) => ({ label: evalDisplayName(d.name, d.file) }))}
            active={active}
            onChange={setActive}
          />
        </Measure>
      )}



      {stats && (
        <Measure>
          <div className="meta-grid">
            <StatCard
              label="Clean macro accuracy"
              value={stats.clean != null ? `${pct(stats.clean)}%` : "–"}
              sub={
                ds?.per_distorted?.clean?.n != null
                  ? `${ds.per_distorted.clean.n.toLocaleString()} clean images`
                  : ds
                    ? `${ds.metrics.n?.toLocaleString() ?? "–"} images`
                    : undefined
              }
            />
            <StatCard
              label="Worst distortion"
              value={stats.worst ? perturbationLabel(stats.worst[0]) : "–"}
              sub={
                stats.worst ? (
                  <span>
                    {pct(stats.worst[1].macro_accuracy)}%
                    {stats.clean != null && (
                      <>
                        {" "}
                        <DeltaChip v={(stats.worst[1].macro_accuracy ?? 0) - stats.clean} />
                      </>
                    )}
                  </span>
                ) : undefined
              }
            />
            <StatCard
              label="Mean degradation"
              value={stats.meanDelta != null ? `${(stats.meanDelta * 100).toFixed(2)} pp` : "–"}
              sub="macro accuracy vs clean, labeled distortions"
            />
            <StatCard
              label="Worst FPR"
              value={`${pct(stats.worstFpr)}%`}
              sub="real images called AI, worst labeled distortion"
            />
          </div>
        </Measure>
      )}

      {ds?.sweep && <SweepTable sweep={ds.sweep} />}

      {ds?.per_distorted && <DistortedPanel ds={ds} />}

      {data && !ds && (
        <Measure>
          <Notice>No robustness tables in this eval dump.</Notice>
        </Measure>
      )}
    </div>
  );
}

function DistortedPanel({ ds }: { ds: EvalDataset }) {
  const clean = ds.per_distorted?.["clean"];
  const distorted = ds.per_distorted?.["distorted"];
  if (!clean || !distorted) return null;
  return (
    <section className="space-y-6">
      <Measure>
        <h2 className="essay-title">Clean vs distorted labels</h2>
        <p className="mt-3 text-[16px] leading-[1.5] text-ink-body">
          NTIRE protocol — images tagged clean or distorted in the public test.
        </p>
        <div className="meta-grid mt-4">
          <div className="meta-pair">
            <dt>Clean images</dt>
            <dd>
              <div className="tabular">{pct(clean.macro_accuracy)}%</div>
              <div className="meta-sub">
                AUROC {pct(clean.auroc)}% · n={clean.n?.toLocaleString() ?? "–"}
              </div>
            </dd>
          </div>
          <div className="meta-pair">
            <dt>Distorted images</dt>
            <dd>
              <div className="tabular">
                {pct(distorted.macro_accuracy)}%
                {clean.macro_accuracy != null && distorted.macro_accuracy != null && (
                  <>
                    {" "}
                    <DeltaChip v={distorted.macro_accuracy - clean.macro_accuracy} />
                  </>
                )}
              </div>
              <div className="meta-sub">
                AUROC {pct(distorted.auroc)}% · n={distorted.n?.toLocaleString() ?? "–"}
                {ds.robust_n != null && ` · robust AUROC ${pct(ds.robust_auroc)}%`}
              </div>
            </dd>
          </div>
        </div>
      </Measure>

      {ds.per_distortion && Object.keys(ds.per_distortion).length > 0 && (
        <div className="figure overflow-x-auto">
          <p className="small-head mb-2">By first distortion type</p>
          <table className="paper-table min-w-[560px]">
            <thead>
              <tr>
                <th>Distortion</th>
                <th>n</th>
                <th>Macro acc</th>
                <th>FPR</th>
                <th>FNR</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(ds.per_distortion)
                .sort((a, b) => (b[1].macro_accuracy ?? 0) - (a[1].macro_accuracy ?? 0))
                .map(([k, m]) => (
                  <tr key={k}>
                    <td>{perturbationLabel(k)}</td>
                    <td className="tabular text-ink-mute">
                      {m.n?.toLocaleString() ?? "–"}
                    </td>
                    <td className="tabular text-ink-head">{pct(m.macro_accuracy)}%</td>
                    <td className="tabular">{pct(m.fpr)}%</td>
                    <td className="tabular">{pct(m.fnr)}%</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
