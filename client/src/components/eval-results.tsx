import { DeltaChip, StatCard } from "@/components/charts";
import { EvalBreakdowns } from "@/components/eval-breakdowns";
import { HeldoutTable } from "@/components/heldout-table";

/**
 * Seer's own held-out numbers — from docs/deliverables/heldout-eval-step27500.md
 * (runs/seer_vitl/last.pt, EMA, step 27,500 of 60,000, clean protocol,
 * threshold 0.5). Pangram's published CompEval numbers are kept as reference
 * deltas, not as the headline.
 */
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
      <HeldoutTable />

      <p className="max-w-3xl text-xs leading-relaxed text-zinc-500">
        Against Pangram Image (CompEval macro acc 97.29% / mAP 99.70%), Seer at
        step 27,500 is 1.64 / 0.08 points behind — with a lower false-positive
        rate. The gap is almost all <span className="text-zinc-300">false
        negatives</span>, concentrated on pixel-space diffusion and a handful of
        stylized frontier generators. Full report:{" "}
        <span className="text-zinc-400">docs/deliverables/heldout-eval-step27500.md</span>
      </p>

      {/* per-set detail, tabbed */}
      <div>
        <h3 className="text-sm font-semibold text-white">Held-out breakdowns</h3>
        <p className="mb-4 mt-1 text-xs text-zinc-500">
          Where the misses live — by family, generator, and source
        </p>
        <EvalBreakdowns />
      </div>
    </div>
  );
}
