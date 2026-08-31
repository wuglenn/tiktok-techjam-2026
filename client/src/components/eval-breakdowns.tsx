"use client";

import { useState } from "react";

import { MetricBar } from "@/components/charts";
import { Measure, Tabs } from "@/components/essay";
import { evalKey, familyLabel, finite, MIRAGE_META } from "@/lib/eval-labels";
import { pct } from "@/lib/format";
import type { EvalDataset } from "@/lib/types";

const TABS = [
  { label: "Community Forensics", hint: "CommunityForensics-Eval by architecture family" },
  { label: "OpenFake", hint: "core/test, fake-only buckets" },
  { label: "MIRAGE", hint: "human-verified in-the-wild" },
] as const;

function findSet(datasets: EvalDataset[], key: string): EvalDataset | undefined {
  return datasets.find((d) => evalKey(d.name, d.file) === key);
}

export function EvalBreakdowns({ datasets }: { datasets: EvalDataset[] }) {
  const [tab, setTab] = useState(0);
  const comfor = findSet(datasets, "comfor_eval");
  const openfake = findSet(datasets, "openfake_test");
  const mirage = findSet(datasets, "mirage");

  return (
    <div>
      <div className="figure">
        <Tabs
          items={TABS.map((t) => ({ label: t.label, hint: t.hint }))}
          active={tab}
          onChange={setTab}
        />
      </div>
      <div className="mt-4">
        {tab === 0 && <CompforFamilies ds={comfor} />}
        {tab === 1 && <OpenfakeGenerators ds={openfake} />}
        {tab === 2 && <MirageSources ds={mirage} />}
      </div>
    </div>
  );
}

function CompforFamilies({ ds }: { ds?: EvalDataset }) {
  const rows = Object.entries(ds?.per_architecture ?? {}).sort(
    (a, b) => (b[1].macro_accuracy ?? 0) - (a[1].macro_accuracy ?? 0),
  );
  if (!rows.length) {
    return <p className="measure caption">CommunityForensics breakdown not in this eval dump.</p>;
  }
  return (
    <div className="figure overflow-x-auto">
      <table className="paper-table min-w-[560px]">
        <thead>
          <tr>
            <th>Family</th>
            <th>n</th>
            <th>Acc</th>
            <th>Recall</th>
            <th>mAP</th>
            <th>FPR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, f]) => (
            <tr key={key}>
              <td className="text-ink-head">{familyLabel(key)}</td>
              <td className="tabular">{f.n?.toLocaleString("en-US")}</td>
              <td>
                <div className="flex items-center gap-2.5">
                  <span className="tabular w-12 text-ink-head">{pct(f.macro_accuracy ?? f.accuracy)}</span>
                  <div className="w-16">
                    <MetricBar value={f.macro_accuracy ?? f.accuracy} />
                  </div>
                </div>
              </td>
              <td className="tabular">{pct(f.recall)}</td>
              <td className="tabular">{finite(f.mAP as number | undefined) != null ? pct(f.mAP as number) : "—"}</td>
              <td className="tabular">{pct(f.fpr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OpenfakeGenerators({ ds }: { ds?: EvalDataset }) {
  const buckets = Object.entries(ds?.per_architecture ?? {});
  const reals = buckets.filter(([, m]) => (m.n_real ?? 0) > 0 && (m.n_fake ?? 0) === 0);
  const fakes = buckets
    .filter(([, m]) => (m.n_fake ?? 0) > 0)
    .sort((a, b) => (b[1].recall ?? 0) - (a[1].recall ?? 0));
  if (!fakes.length) {
    return <p className="measure caption">OpenFake generator breakdown not in this eval dump.</p>;
  }
  const docci = reals.find(([k]) => k === "docci")?.[1];
  const imagenet = reals.find(([k]) => k === "imagenet")?.[1];
  const hole = fakes.filter(([, m]) => (m.recall ?? 1) < 0.86).map(([k]) => k);
  return (
    <div>
      <div className="figure overflow-x-auto">
        <table className="paper-table min-w-[520px]">
          <thead>
            <tr>
              <th>Generator</th>
              <th>n</th>
              <th>Recall</th>
              <th>FNR</th>
            </tr>
          </thead>
          <tbody>
            {fakes.map(([gen, g]) => (
              <tr key={gen}>
                <td className="text-ink-head">{gen}</td>
                <td className="tabular">{g.n?.toLocaleString("en-US")}</td>
                <td>
                  <div className="flex items-center gap-2.5">
                    <span className="tabular w-12 text-ink-head">{pct(g.recall)}</span>
                    <div className="w-20">
                      <MetricBar value={g.recall} />
                    </div>
                  </div>
                </td>
                <td className="tabular">{pct(g.fnr ?? (g.recall != null ? 1 - g.recall : undefined))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Measure className="mt-3">
        <p className="text-[16px] leading-[1.5] text-ink-body">
          Fake-only buckets — no reals per bucket, so precision is 1.0 and
          accuracy equals recall. Held-out reals: DOCCI{" "}
          {docci ? `${pct(docci.fpr)}% FPR (${docci.n?.toLocaleString("en-US")} images)` : "—"}{" "}
          and ImageNet{" "}
          {imagenet ? `${pct(imagenet.fpr)}% FPR (${imagenet.n?.toLocaleString("en-US")})` : "—"}.
          {hole.length > 0 && (
            <>
              {" "}
              The hole is a small set of stylized generators — {hole.join(", ")} — not
              the GPT Image / FLUX.2 / Seedream mass.
            </>
          )}
        </p>
      </Measure>
    </div>
  );
}

function MirageSources({ ds }: { ds?: EvalDataset }) {
  const rows = Object.entries(ds?.per_architecture ?? {}).sort(
    (a, b) => (b[1].recall ?? 0) - (a[1].recall ?? 0),
  );
  if (!rows.length) {
    return <p className="measure caption">MIRAGE source breakdown not in this eval dump.</p>;
  }
  const full = rows.filter(([k]) => k === "T2I" || k === "RMG" || k === "PCRMG");
  const local = rows.filter(([k]) => ["IE", "IP/OP", "FS", "TR", "CB"].includes(k));
  const mixed = rows.filter(([k]) => k === "IID" || k === "OOD-R");
  return (
    <div>
      <div className="figure overflow-x-auto">
        <table className="paper-table min-w-[560px]">
          <thead>
            <tr>
              <th>Source</th>
              <th>n (fake / real)</th>
              <th>Acc</th>
              <th>Recall</th>
              <th>FPR</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, s]) => {
              const meta = MIRAGE_META[key];
              const acc = s.n_real ? s.macro_accuracy ?? s.accuracy : s.recall ?? s.accuracy;
              return (
                <tr key={key}>
                  <td>
                    <div className="text-ink-head">
                      {meta?.name ?? key}{" "}
                      <span className="font-mono text-[12px] text-ink-mute">{key}</span>
                    </div>
                    {meta && (
                      <div className="mt-0.5 max-w-[280px] font-mono text-[12px] leading-snug text-ink-mute">
                        {meta.desc}
                      </div>
                    )}
                  </td>
                  <td className="tabular">
                    {(s.n_fake ?? 0).toLocaleString("en-US")} / {(s.n_real ?? 0).toLocaleString("en-US")}
                  </td>
                  <td>
                    <div className="flex items-center gap-2.5">
                      <span className="tabular w-12 text-ink-head">{pct(acc)}</span>
                      <div className="w-16">
                        <MetricBar value={acc} />
                      </div>
                    </div>
                  </td>
                  <td className="tabular">{pct(s.recall)}</td>
                  <td className="tabular">
                    {s.n_real ? pct(s.fpr) : <span className="text-ink-mute">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Measure className="mt-3">
        <p className="text-[16px] leading-[1.5] text-ink-body">
          MIRAGE tags each image by how it was built, not by which generator
          built it. Full-image synthesis is largely solved
          {full.length > 0 && (
            <>
              {" "}
              — {full.map(([k, m]) => `${k} ${pct(m.recall)}%`).join(", ")}
            </>
          )}
          . The hole is local edits
          {local.length > 0 && (
            <>
              {" "}
              ({local.map(([k, m]) => `${k} ${pct(m.recall)}%`).join(", ")})
            </>
          )}
          — exactly the composite-like edits the patch head is meant to catch,
          but this pass is page-level only.
          {mixed.length > 0 && (
            <>
              {" "}
              The mixed real/fake slices are the human-curated ones
              {mixed.map(([, m]) => `, ${pct(m.fpr)}% FPR`).join("")}.
            </>
          )}
        </p>
      </Measure>
    </div>
  );
}

