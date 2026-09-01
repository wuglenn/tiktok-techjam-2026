import { NextResponse } from "next/server";

import {
  findCheckpoint,
  findPython,
  modalServerUrl,
  probeInferServer,
  repoRoot,
  runBridge,
  runModal,
} from "@/lib/seer-server";
import type { UploadFile } from "@/lib/seer-server";
import type { AnalyzeResponse } from "@/lib/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 300;

const MAX_FILES = 12;
const MAX_BYTES = 40 * 1024 * 1024;

/** localhost / loopback hosts — the only case where the local model is used. */
function isLocalhost(req: Request): boolean {
  const host = (req.headers.get("host") || new URL(req.url).host).toLowerCase();
  const hostname = host.replace(/:\d+$/, "").replace(/^\[|\]$/g, "");
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

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

  // Backend selection: the local model when the dashboard itself runs on
  // localhost and one is available; the Modal deployment otherwise. There
  // is no simulated fallback — if nothing works, that is an error.
  const started = Date.now();
  if (isLocalhost(req)) {
    const root = repoRoot();
    const server = await probeInferServer();
    const checkpoint = server?.checkpoint ?? (root ? findCheckpoint(root) : null);
    const interpreter = root ? findPython(root) : null;
    if (root && (server?.ok || (checkpoint && interpreter)) && checkpoint) {
      try {
        const results = await runBridge(root, checkpoint, files);
        const elapsed = Date.now() - started;
        const body: AnalyzeResponse = {
          backend: "local",
          checkpoint,
          results: results.map((r) => ({ ...r, elapsedMs: Math.round(elapsed / results.length) })),
        };
        return NextResponse.json(body);
      } catch (err) {
        const note = err instanceof Error ? err.message : String(err);
        return NextResponse.json(
          {
            error: `local inference failed — ${note.slice(0, 400)}${note.length > 400 ? "…" : ""}`,
          },
          { status: 502 },
        );
      }
    }
  }

  const modal = modalServerUrl();
  if (!modal) {
    return NextResponse.json(
      {
        error:
          "no inference backend — deploy one with `modal deploy client/scripts/modal_seer.py` and set SEER_MODAL_URL, or run client/scripts/seer_serve.py locally",
      },
      { status: 503 },
    );
  }
  try {
    const results = await runModal(files);
    const elapsed = Date.now() - started;
    const body: AnalyzeResponse = {
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
