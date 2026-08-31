import { NextResponse } from "next/server";

import { findCheckpoint, repoRoot, runBridge, simulateFile } from "@/lib/seer-server";
import type { UploadFile } from "@/lib/seer-server";
import type { AnalyzeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 300;

const MAX_FILES = 12;
const MAX_BYTES = 40 * 1024 * 1024;

export async function POST(req: Request) {
  let files: UploadFile[] = [];
  try {
    const form = await req.formData();
    for (const f of form.getAll("files")) {
      if (!(f instanceof File)) continue;
      if (!f.type.startsWith("image/")) {
        return NextResponse.json(
          { error: `"${f.name}" is not an image (got ${f.type || "unknown type"})` },
          { status: 400 },
        );
      }
      files.push({ name: f.name, type: f.type, buf: Buffer.from(await f.arrayBuffer()) });
    }
  } catch {
    return NextResponse.json({ error: "could not read upload (multipart form expected)" }, { status: 400 });
  }

  files = files.slice(0, MAX_FILES);
  if (!files.length) {
    return NextResponse.json({ error: "no image files received" }, { status: 400 });
  }
  if (files.some((f) => f.buf.length > MAX_BYTES)) {
    return NextResponse.json({ error: "one of the files exceeds the 40 MB limit" }, { status: 400 });
  }

  const root = repoRoot();
  const checkpoint = root ? findCheckpoint(root) : null;
  const t0 = Date.now();

  if (root && checkpoint) {
    try {
      const results = await runBridge(root, checkpoint, files);
      const elapsed = Date.now() - t0;
      const body: AnalyzeResponse = {
        mode: "live",
        checkpoint,
        results: results.map((r) => ({ ...r, elapsedMs: Math.round(elapsed / results.length) })),
      };
      return NextResponse.json(body);
    } catch (err) {
      const note = err instanceof Error ? err.message : String(err);
      const body: AnalyzeResponse = {
        mode: "simulated",
        checkpoint,
        note: `live inference failed — ${note.slice(0, 400)}${note.length > 400 ? "…" : ""}`,
        results: files.map(simulateFile),
      };
      return NextResponse.json(body);
    }
  }

  const body: AnalyzeResponse = {
    mode: "simulated",
    checkpoint,
    note: root
      ? "no checkpoint found under runs/*/best.pt (train one, or point SEER_CHECKPOINT at a .pt file) — showing deterministic simulated verdicts"
      : "Seer repo root not found relative to the dashboard — showing deterministic simulated verdicts",
    results: files.map(simulateFile),
  };
  return NextResponse.json(body);
}
