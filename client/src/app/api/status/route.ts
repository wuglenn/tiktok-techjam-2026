import { NextResponse } from "next/server";

import { findCheckpoint, findPython, repoRoot } from "@/lib/seer-server";
import type { StatusResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const root = repoRoot();
  const checkpoint = root ? findCheckpoint(root) : null;
  const uv = root ? findPython(root) : null;
  const body: StatusResponse = {
    mode: checkpoint && uv ? "live" : "simulated",
    checkpoint,
    root,
    uv,
  };
  return NextResponse.json(body);
}
