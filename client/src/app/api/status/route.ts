import { NextResponse } from "next/server";

import {
  findCheckpoint,
  findPython,
  inferServerUrl,
  modalServerUrl,
  probeInferServer,
  repoRoot,
} from "@/lib/seer-server";
import type { StatusResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Short on purpose: a cold Modal container would otherwise boot on the probe. */
const MODAL_PROBE_MS = 2500;

export async function GET() {
  const root = repoRoot();
  const server = await probeInferServer();

  const modalUrl = modalServerUrl();
  let modal: StatusResponse["modal"] = null;
  if (modalUrl) {
    const health = await probeInferServer(modalUrl, MODAL_PROBE_MS);
    modal = {
      url: modalUrl,
      // null = unreachable within the probe window (cold or down); the
      // first analyze request would boot it
      ok: health ? Boolean(health.ok) : null,
      device: health?.device ?? null,
      checkpoint: health?.checkpoint ?? null,
      error: health && !health.ok ? (health.error ?? "not ready") : null,
    };
  }

  const localCheckpoint = root ? findCheckpoint(root) : null;
  const modalLive = modal?.ok === true;
  const checkpoint =
    server?.checkpoint ?? localCheckpoint ?? modal?.checkpoint ?? null;
  const uv = root ? findPython(root) : null;
  const live =
    Boolean(server?.ok) || Boolean(localCheckpoint && uv && !server) || modalLive;
  const body: StatusResponse = {
    mode: live ? "live" : "simulated",
    checkpoint,
    root,
    uv,
    server: server ? inferServerUrl() : null,
    device: server?.device ?? (modalLive && !localCheckpoint ? (modal?.device ?? null) : null),
    error: server && !server.ok ? (server.error ?? "model is still loading") : null,
    modal,
  };
  return NextResponse.json(body);
}
