import { DeltaChip, StatCard } from "@/components/charts";
import { EvalBreakdowns } from "@/components/eval-breakdowns";
import { HeldoutTable } from "@/components/heldout-table";

/**
 * Seer's own held-out numbers — from docs/deliverables/heldout-eval-step27500.md
 * (runs/seer_vitl/last.pt, EMA, step 27,500 of 60,000, clean protocol,
 * threshold 0.5). Pangram's published CommunityForensics-Eval numbers are
 * kept as reference deltas, not as the headline.
 */
export function EvalResults() {
  return (
    <div className="mt-8 space-y-8">
      {/* held-out results */}
      <HeldoutTable />

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
