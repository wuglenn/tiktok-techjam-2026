import { NextResponse } from "next/server";

import {
  findCheckpoint,
  modalServerUrl,
  repoRoot,
  runBridge,
  runModal,
  simulateFile,
} from "@/lib/seer-server";
import type { UploadFile } from "@/lib/seer-server";
import type { AnalyzeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 300;

const MAX_FILES = 12;
const MAX_BYTES = 40 * 1024 * 1024;

export async function POST(req: Request) {
  let files: UploadFile[] = [];
  let useModal = false;
  try {
    const form = await req.formData();
    useModal =
      form.get("backend") === "modal" ||
      new URL(req.url).searchParams.get("backend") === "modal";
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

  // Modal deployment flag ("Score on Modal" toggle on /analyze): score
  // remotely, no local checkpoint / interpreter / repo root required.
  if (useModal) {
    const modal = modalServerUrl();
    if (!modal) {
      return NextResponse.json(
        {
          error:
            "Modal backend requested but SEER_MODAL_URL is not set — run `modal deploy client/scripts/modal_seer.py` and export SEER_MODAL_URL to the printed URL",
        },
        { status: 400 },
      );
    }
    const started = Date.now();
    try {
      const results = await runModal(files);
      const elapsed = Date.now() - started;
      const body: AnalyzeResponse = {
        mode: "live",
        backend: "modal",
        checkpoint: modal,
        results: results.map((r) => ({ ...r, elapsedMs: Math.round(elapsed / results.length) })),
      };
      return NextResponse.json(body);
    } catch (err) {
      const note = err instanceof Error ? err.message : String(err);
      return NextResponse.json(
        {
          error: `modal inference failed — ${note.slice(0, 400)}${note.length > 400 ? "…" : ""}`,
        },
        { status: 502 },
      );
    }
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
        backend: "local",
        checkpoint,
        results: results.map((r) => ({ ...r, elapsedMs: Math.round(elapsed / results.length) })),
      };
      return NextResponse.json(body);
    } catch (err) {
      const note = err instanceof Error ? err.message : String(err);
      return NextResponse.json(
        {
          error: `live inference failed — ${note.slice(0, 400)}${note.length > 400 ? "…" : ""}`,
        },
        { status: 502 },
      );
    }
  }

  const modal = modalServerUrl();
  const body: AnalyzeResponse = {
    mode: "simulated",
    checkpoint,
    note: modal
      ? "no local checkpoint — tick “Score on Modal” to run inference on the remote deployment"
      : root
        ? "no checkpoint found (repo-root best.pt or runs/*/best.pt). Start client/scripts/seer_serve.py or set SEER_CHECKPOINT — showing deterministic simulated verdicts"
        : "Seer repo root not found relative to the dashboard — showing deterministic simulated verdicts",
    results: files.map(simulateFile),
  };
  return NextResponse.json(body);
}
