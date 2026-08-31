import { NextResponse } from "next/server";

import {
  findCheckpoint,
  findPython,
  inferServerUrl,
  probeInferServer,
  repoRoot,
} from "@/lib/seer-server";
import type { StatusResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const root = repoRoot();
  const server = await probeInferServer();
  const checkpoint =
    server?.checkpoint ?? (root ? findCheckpoint(root) : null);
  const uv = root ? findPython(root) : null;
  const live = Boolean(server?.ok) || Boolean(checkpoint && uv && !server);
  const body: StatusResponse = {
    mode: live ? "live" : "simulated",
    checkpoint,
    root,
    uv,
    server: server ? inferServerUrl() : null,
    device: server?.device ?? null,
    error: server && !server.ok ? (server.error ?? "model is still loading") : null,
  };
  return NextResponse.json(body);
}
