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
      ? "no eval JSONs under eval/eval_step33500 or runs/ — drop a seer/eval.py --out-json dump there to replace the placeholders"
      : "Seer repo root not found — showing bundled demo data",
  };
  return NextResponse.json(body);
}
