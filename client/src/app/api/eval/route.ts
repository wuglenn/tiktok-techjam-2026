import { NextResponse } from "next/server";

import { DEMO_DATASETS } from "@/lib/demo-data";
import { repoRoot, scanEvalRuns } from "@/lib/seer-server";
import type { EvalResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const root = repoRoot();
  if (root) {
    const datasets = scanEvalRuns(root);
    if (datasets.length) {
      const body: EvalResponse = { mode: "live", root, datasets };
      return NextResponse.json(body);
    }
  }
  const body: EvalResponse = {
    mode: "demo",
    root,
    datasets: DEMO_DATASETS,
    note: root
      ? "no eval JSONs under runs/ yet — run `uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val --perturbation all --error-dir runs/eval/errors --out-json runs/eval/ntire_val.json` to populate this page with real numbers"
      : "Seer repo root not found — showing bundled demo data",
  };
  return NextResponse.json(body);
}
