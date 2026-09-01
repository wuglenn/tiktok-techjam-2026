import fs from "node:fs";

import { NextRequest, NextResponse } from "next/server";

import { repoRoot, safeEvalImagePath } from "@/lib/seer-server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
};

/** Serves error-analysis panels written by `eval.py --error-dir`. */
export async function GET(req: NextRequest) {
  const rel = req.nextUrl.searchParams.get("src");
  if (!rel) return new NextResponse("missing src", { status: 400 });

  const root = repoRoot();

  const abs = safeEvalImagePath(root, rel);
  if (!abs) return new NextResponse("not found", { status: 404 });

  const ext = abs.split(".").pop()!.toLowerCase();
  const buf = fs.readFileSync(abs);
  return new NextResponse(new Uint8Array(buf), {
    headers: {
      "content-type": TYPES[ext] ?? "application/octet-stream",
      "cache-control": "public, max-age=3600, immutable",
    },
  });
}
