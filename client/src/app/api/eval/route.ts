import { NextResponse } from "next/server";

import { DEMO_DATASETS } from "@/lib/demo-data";
import { repoRoot, scanEvalRuns } from "@/lib/seer-server";
import type { EvalResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const root = repoRoot();
  // the committed suite is bundled in client/eval, so this works even
  // without the Python repo (e.g. dashboard served with the Modal backend)
  const datasets = scanEvalRuns(root);
  if (datasets.length) {
    const body: EvalResponse = { mode: "live", root, datasets };
    return NextResponse.json(body);
  }
  const body: EvalResponse = {
    mode: "demo",
    root,
    datasets: DEMO_DATASETS,
    note: root
      ? "no eval JSONs under client/eval/eval_step33500 or runs/ — drop a seer/eval.py --out-json dump there to replace the placeholders"
      : "standalone dashboard (no Seer repo root) — showing bundled demo data",
  };
  return NextResponse.json(body);
}
