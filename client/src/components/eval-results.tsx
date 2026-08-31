import { DeltaChip, MetricBar, StatCard } from "@/components/charts";
import { pct } from "@/lib/format";

/**
 * Seer's own held-out numbers — from docs/deliverables/heldout-eval-step27500.md
 * (runs/seer_vitl/last.pt, EMA, step 27,500 of 60,000, clean protocol,
 * threshold 0.5). Pangram's published CompEval numbers are kept as reference
 * deltas, not as the headline.
 */

interface HeldoutRow {
  set: string;
  n: string;
  split: string;
  macro?: number;
  map?: number;
  auroc?: number;
  f1?: number;
  fpr?: number;
  fnr?: number;
}

const HELDOUT: HeldoutRow[] = [
  {
    set: "CommunityForensics-Eval",
    n: "51,836",
    split: "25,918 / 25,918",
    macro: 0.9565,
    map: 0.9962,
    auroc: 0.9954,
    f1: 0.9546,
    fpr: 0.0018,
    fnr: 0.0852,
  },
  {
    set: "OpenFake core/test",
    n: "89,225",
    split: "45,697 / 43,528",
    macro: 0.9719,
    map: 0.9984,
    auroc: 0.9981,
    f1: 0.9712,
    fpr: 0.0021,
    fnr: 0.0542,
  },
  {
    set: "OpenFake reddit/test",
    n: "36,227",
    split: "29,116 / 7,111",
    macro: 0.8905,
    map: 0.9928,
    auroc: 0.9728,
    f1: 0.8874,
    fpr: 0.0205,
    fnr: 0.1984,
  },
  {
    set: "MIRAGE",
    n: "12,073",
    split: "10,682 / 1,391",
    macro: 0.8626,
    map: 0.9902,
    auroc: 0.9302,
    f1: 0.8732,
    fpr: 0.0554,
    fnr: 0.2194,
  },
  {
    set: "COCO val2017 (reals only)",
    n: "5,000",
    split: "0 / 5,000",
    fpr: 0.001,
  },
];

const COMPFOR_FAMILIES = [
  { family: "Other", n: "2,000", acc: 0.999, recall: 0.998, map: 1.0, fpr: 0.0 },
  { family: "Latent diffusion", n: "12,000", acc: 0.9968, recall: 0.995, map: 1.0, fpr: 0.0013 },
  { family: "GAN", n: "4,000", acc: 0.994, recall: 0.989, map: 1.0, fpr: 0.001 },
  { family: "Commercial", n: "29,836", acc: 0.9444, recall: 0.8912, map: 0.9955, fpr: 0.0024 },
  { family: "Pixel diffusion", n: "4,000", acc: 0.867, recall: 0.7345, map: 0.9884, fpr: 0.0005 },
];

const WORST_GENERATORS = [
  { name: "recraft-v3", recall: "56.9%" },
  { name: "halfmoon-4-4-25", recall: "72.1%" },
  { name: "frames-23-1-25", recall: "75.6%" },
  { name: "ideogram-2.0", recall: "75.9%" },
  { name: "midjourney-7", recall: "83.3%" },
];

export function EvalResults() {
  return (
    <div className="mt-8 space-y-8">
      {/* headline cards — Pangram as the reference delta, not the headline */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="CompEval macro acc"
          value="95.65%"
          sub={
            <span className="flex items-center gap-1.5">
              Pangram 97.29%
              <DeltaChip v={-0.0164} vs="Pangram" />
            </span>
          }
          accent="cyan"
        />
        <StatCard
          label="CompEval mAP"
          value="99.62%"
          sub={
            <span className="flex items-center gap-1.5">
              Pangram 99.70%
              <DeltaChip v={-0.0008} vs="Pangram" />
            </span>
          }
          accent="cyan"
        />
        <StatCard
          label="CompEval FPR"
          value="0.18%"
          sub="lower than Pangram's careful-FPR pitch"
          accent="emerald"
        />
        <StatCard
          label="COCO val2017 FPR"
          value="0.10%"
          sub="5 false positives on 5,000 photos"
          accent="emerald"
        />
      </div>

      {/* held-out results */}
      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                <th className="px-5 py-3.5 font-medium">Held-out set</th>
                <th className="px-4 py-3.5 font-medium">n (fake / real)</th>
                <th className="px-4 py-3.5 font-medium">Macro acc</th>
                <th className="px-4 py-3.5 font-medium">mAP</th>
                <th className="px-4 py-3.5 font-medium">AUROC</th>
                <th className="px-4 py-3.5 font-medium">F1</th>
                <th className="px-4 py-3.5 font-medium">FPR</th>
                <th className="px-5 py-3.5 font-medium">FNR</th>
              </tr>
            </thead>
            <tbody>
              {HELDOUT.map((r, i) => (
                <tr
                  key={r.set}
                  className={`transition-colors hover:bg-white/[0.03] ${
                    i < HELDOUT.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <td className="px-5 py-3.5 font-medium text-zinc-100">{r.set}</td>
                  <td className="tabular px-4 py-3.5 text-zinc-400">
                    <span className="text-zinc-300">{r.n}</span>
                    <span className="text-zinc-500"> ({r.split})</span>
                  </td>
                  <td className="px-4 py-3.5">
                    {r.macro != null ? (
                      <div className="flex items-center gap-2.5">
                        <span className="tabular w-12 font-medium text-white">
                          {pct(r.macro)}
                        </span>
                        <div className="w-20">
                          <MetricBar value={r.macro} />
                        </div>
                      </div>
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                  <td className="tabular px-4 py-3.5 text-zinc-300">
                    {r.map != null ? pct(r.map) : "—"}
                  </td>
                  <td className="tabular px-4 py-3.5 text-zinc-300">
                    {r.auroc != null ? pct(r.auroc) : "—"}
                  </td>
                  <td className="tabular px-4 py-3.5 text-zinc-300">
                    {r.f1 != null ? pct(r.f1) : "—"}
                  </td>
                  <td className="tabular px-4 py-3.5 text-emerald-300/90">
                    {r.fpr != null ? pct(r.fpr) : "—"}
                  </td>
                  <td className="tabular px-5 py-3.5 text-amber-300/90">
                    {r.fnr != null ? pct(r.fnr) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="max-w-3xl text-xs leading-relaxed text-zinc-500">
        Against Pangram Image (CompEval macro acc 97.29% / mAP 99.70%), Seer at
        step 27,500 is 1.64 / 0.08 points behind — with a lower false-positive
        rate. The gap is almost all <span className="text-zinc-300">false
        negatives</span>, concentrated on pixel-space diffusion and a handful of
        stylized frontier generators. Full report:{" "}
        <span className="text-zinc-400">docs/deliverables/heldout-eval-step27500.md</span>
      </p>

      {/* CompEval by generator family */}
      <div>
        <h3 className="text-sm font-semibold text-white">
          CommunityForensics-Eval by generator family
        </h3>
        <div className="panel mt-3 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-sm">
              <thead>
                <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                  <th className="px-5 py-3 font-medium">Family</th>
                  <th className="px-4 py-3 font-medium">n</th>
                  <th className="px-4 py-3 font-medium">Acc</th>
                  <th className="px-4 py-3 font-medium">Recall</th>
                  <th className="px-4 py-3 font-medium">mAP</th>
                  <th className="px-5 py-3 font-medium">FPR</th>
                </tr>
              </thead>
              <tbody>
                {COMPFOR_FAMILIES.map((f, i) => (
                  <tr
                    key={f.family}
                    className={`transition-colors hover:bg-white/[0.03] ${
                      i < COMPFOR_FAMILIES.length - 1
                        ? "border-b border-white/[0.04]"
                        : ""
                    }`}
                  >
                    <td className="px-5 py-3 font-medium text-zinc-100">{f.family}</td>
                    <td className="tabular px-4 py-3 text-zinc-400">{f.n}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        <span className="tabular w-12 font-medium text-white">
                          {pct(f.acc)}
                        </span>
                        <div className="w-20">
                          <MetricBar value={f.acc} />
                        </div>
                      </div>
                    </td>
                    <td className="tabular px-4 py-3 text-zinc-300">{pct(f.recall)}</td>
                    <td className="tabular px-4 py-3 text-zinc-300">{pct(f.map)}</td>
                    <td className="tabular px-5 py-3 text-emerald-300/90">{pct(f.fpr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p className="mt-3 max-w-3xl text-xs leading-relaxed text-zinc-500">
          Open generators (GAN, latent diffusion, other) are essentially solved.
          The gap sits in commercial and pixel-space diffusion — pixel-space
          still ranks high (mAP 98.84%) but 26.6% of its fakes fall under the
          0.5 threshold.
        </p>
      </div>

      {/* hardest held-out generators */}
      <div>
        <h3 className="text-sm font-semibold text-white">
          Hardest held-out generators
        </h3>
        <p className="mt-1 text-xs text-zinc-500">
          OpenFake core/test, fake-only recall at threshold 0.5
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {WORST_GENERATORS.map((g) => (
            <span
              key={g.name}
              className="inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs text-zinc-300"
            >
              {g.name}
              <span className="tabular font-semibold text-rose-300">{g.recall}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
