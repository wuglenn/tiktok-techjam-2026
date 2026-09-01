/**
 * Server-only glue between the dashboard and the Seer Python package.
 *
 * Live inference runs `scripts/seer_infer.py` through the repo's uv
 * environment (or `.venv` python); when no checkpoint / interpreter is
 * available the API falls back to a deterministic simulation so the demo
 * still works end-to-end.
 */
import { spawn, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { hashBytes, syntheticGrid } from "./heat";
import type { AnalyzeResult, ErrorEntry, EvalDataset, MetricsRow } from "./types";

/** Walk up from cwd until the Seer repo root (has src/seer + pyproject). */
export function repoRoot(): string | null {
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    if (
      fs.existsSync(path.join(dir, "src", "seer", "model.py")) &&
      fs.existsSync(path.join(dir, "pyproject.toml"))
    ) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

/**
 * Best checkpoint to serve: $SEER_CHECKPOINT, else repo-root `best.pt`
 * (the hero weights `predict.py` defaults to), else the newest
 * `runs/<name>/best.pt` (preferring the hero `seer_vitl*` runs).
 */
export function findCheckpoint(root: string): string | null {
  const env = process.env.SEER_CHECKPOINT;
  if (env) {
    const p = path.isAbsolute(env) ? env : path.join(root, env);
    if (fs.existsSync(p)) return p;
  }
  const rootCkpt = path.join(root, "best.pt");
  if (fs.existsSync(rootCkpt)) return rootCkpt;
  const runsDir = path.join(root, "runs");
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(runsDir, { withFileTypes: true });
  } catch {
    return null;
  }
  const cands: { p: string; mtime: number; hero: number }[] = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const ckpt = path.join(runsDir, e.name, "best.pt");
    try {
      const st = fs.statSync(ckpt);
      cands.push({
        p: ckpt,
        mtime: st.mtimeMs,
        hero: /seer_vit[lh]/i.test(e.name) ? 1 : 0,
      });
    } catch {
      /* no best.pt in this run */
    }
  }
  if (!cands.length) return null;
  cands.sort((a, b) => b.hero - a.hero || b.mtime - a.mtime);
  return cands[0].p;
}

/** Interpreter candidates for the bridge, in preference order. */
function pythonCandidates(root: string): { cmd: string; prefix: string[] }[] {
  const out: { cmd: string; prefix: string[] }[] = [];
  if (process.env.SEER_PYTHON) out.push({ cmd: process.env.SEER_PYTHON, prefix: [] });
  out.push({ cmd: process.platform === "win32" ? "uv.exe" : "uv", prefix: ["run", "python"] });
  const venvPy = path.join(
    root,
    ".venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
  if (fs.existsSync(venvPy)) out.push({ cmd: venvPy, prefix: [] });
  return out;
}

/** Which interpreter would be used (for /api/status), or null. */
export function findPython(root: string): string | null {
  for (const c of pythonCandidates(root)) {
    if (c.prefix.length === 0) {
      if (fs.existsSync(c.cmd)) return c.cmd;
    } else if (spawnSyncQuick(c.cmd, ["--version"])) {
      return c.cmd;
    }
  }
  return null;
}

function spawnSyncQuick(cmd: string, args: string[]): boolean {
  try {
    const r = spawnSync(cmd, args, { stdio: "ignore", timeout: 10_000, shell: false });
    return !r.error;
  } catch {
    return false;
  }
}

function runCapture(
  cmd: string,
  args: string[],
  opts: { cwd: string; timeout: number },
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd: opts.cwd, shell: false });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        child.kill();
        reject(new Error(`${cmd} timed out after ${opts.timeout / 1000}s`));
      }
    }, opts.timeout);
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("error", (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });
    child.on("close", (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        if (code === 0) resolve({ stdout, stderr, code });
        else reject(new Error(`${cmd} exited with ${code}\n${stderr.slice(-2000)}`));
      }
    });
  });
}

export interface UploadFile {
  name: string;
  type: string;
  buf: Buffer;
}

const EXT_BY_TYPE: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
  "image/bmp": "bmp",
  "image/tiff": "tif",
};

/** Extract intrinsic dimensions from common image headers (no decode). */
export function imageDims(buf: Buffer): { width: number; height: number } | undefined {
  try {
    // PNG
    if (buf.length > 24 && buf.readUInt32BE(0) === 0x89504e47) {
      return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
    }
    // GIF
    if (buf.length > 10 && buf.toString("ascii", 0, 3) === "GIF") {
      return { width: buf.readUInt16LE(6), height: buf.readUInt16LE(8) };
    }
    // WebP (VP8X canvas size)
    if (buf.length > 30 && buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WEBP") {
      const chunk = buf.toString("ascii", 12, 16);
      if (chunk === "VP8X") {
        return {
          width: 1 + (buf[24] | (buf[25] << 8) | (buf[26] << 16)),
          height: 1 + (buf[27] | (buf[28] << 8) | (buf[29] << 16)),
        };
      }
      if (chunk === "VP8 " && buf.length > 38) {
        return {
          width: (buf[26] | (buf[27] << 8)) & 0x3fff,
          height: ((buf[27] >> 6) | (buf[28] << 2) | (buf[29] << 10)) & 0x3fff,
        };
      }
      if (chunk === "VP8L" && buf.length > 25) {
        const b = buf.slice(21, 25);
        return {
          width: 1 + (((b[1] & 0x3f) << 8) | b[0]),
          height: 1 + (((b[3] & 0xf) << 10) | (b[2] << 2) | ((b[1] & 0xc0) >> 6)),
        };
      }
    }
    // JPEG
    if (buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8) {
      let i = 2;
      while (i + 9 < buf.length) {
        if (buf[i] !== 0xff) {
          i++;
          continue;
        }
        const marker = buf[i + 1];
        if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
          return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
        }
        i += 2 + buf.readUInt16BE(i + 2);
      }
    }
  } catch {
    /* best-effort only */
  }
  return undefined;
}

export function inferServerUrl(): string {
  const raw = process.env.SEER_INFER_URL?.trim();
  return (raw || "http://127.0.0.1:8765").replace(/\/$/, "");
}

export interface InferServerHealth {
  ok: boolean;
  ready?: boolean;
  checkpoint?: string;
  device?: string;
  backbone?: string;
  step?: number;
  res?: number;
  error?: string;
}

/** Probe the persistent `seer_serve.py` process. null = nothing listening. */
export async function probeInferServer(
  url = inferServerUrl(),
  timeoutMs = 800,
): Promise<InferServerHealth | null> {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(`${url}/health`, { signal: ctrl.signal, cache: "no-store" });
    clearTimeout(timer);
    const body = (await res.json()) as Record<string, unknown>;
    return {
      ok: Boolean(body.ok),
      ready: Boolean(body.ready),
      checkpoint: typeof body.checkpoint === "string" ? body.checkpoint : undefined,
      device: typeof body.device === "string" ? body.device : undefined,
      backbone: typeof body.backbone === "string" ? body.backbone : undefined,
      step: typeof body.step === "number" ? body.step : undefined,
      res: typeof body.res === "number" ? body.res : undefined,
      error: typeof body.error === "string" ? body.error : undefined,
    };
  } catch {
    return null;
  }
}

type BridgeRecord = {
  image: string;
  prob_ai: number;
  label: string;
  grid: number[][] | null;
  width?: number;
  height?: number;
};

function recordsToResults(records: BridgeRecord[], files: UploadFile[]): AnalyzeResult[] {
  return records.map((r, i) => {
    const dims = files[i] ? imageDims(files[i].buf) : undefined;
    return {
      name: files[i]?.name ?? path.basename(r.image),
      prob_ai: r.prob_ai,
      label: r.label === "AI" ? ("AI" as const) : ("REAL" as const),
      grid: r.grid,
      // prefer the uploaded file's intrinsic size — the model resizes to a
      // square 512 for the forward pass, but the overlay must match the photo
      width: dims?.width ?? r.width,
      height: dims?.height ?? r.height,
      bytes: files[i]?.buf.length,
      type: files[i]?.type,
    };
  });
}

async function runViaServer(
  url: string,
  files: UploadFile[],
  saved: string[],
  timeoutMs: number,
): Promise<AnalyzeResult[]> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${url}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: saved }),
      signal: ctrl.signal,
      cache: "no-store",
    });
    const text = await res.text();
    if (!res.ok) {
      let detail = text.slice(0, 400);
      try {
        const parsed = JSON.parse(text) as { error?: string };
        if (parsed.error) detail = parsed.error;
      } catch {
        /* keep raw text */
      }
      throw new Error(`infer server ${res.status}: ${detail}`);
    }
    const records = JSON.parse(text) as BridgeRecord[];
    if (!Array.isArray(records)) throw new Error("infer server returned no array");
    return recordsToResults(records, files);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Live inference: save uploads to a scratch dir, score them on the
 * persistent server when it is up, otherwise spawn `seer_infer.py`.
 */
export async function runBridge(
  root: string,
  checkpoint: string,
  files: UploadFile[],
  timeoutMs = 300_000,
): Promise<AnalyzeResult[]> {
  const clientDir = process.cwd();
  const script = path.join(clientDir, "scripts", "seer_infer.py");
  const tmp = path.join(clientDir, ".seer-tmp", crypto.randomUUID());
  fs.mkdirSync(tmp, { recursive: true });
  const saved: string[] = [];
  try {
    files.forEach((f, i) => {
      const ext = EXT_BY_TYPE[f.type] ?? "bin";
      const p = path.join(tmp, `img-${i}.${ext}`);
      fs.writeFileSync(p, f.buf);
      saved.push(p);
    });

    const url = inferServerUrl();
    const health = await probeInferServer(url);
    if (health?.ok) {
      return await runViaServer(url, files, saved, timeoutMs);
    }
    if (health && !health.ok) {
      throw new Error(health.error || "inference server is still loading the model");
    }

    let lastErr: unknown = null;
    for (const cand of pythonCandidates(root)) {
      try {
        const { stdout } = await runCapture(
          cand.cmd,
          [...cand.prefix, script, "--checkpoint", checkpoint, "--image", ...saved],
          { cwd: root, timeout: timeoutMs },
        );
        const start = Math.min(
          ...["[\n", "[{"].map((s) => {
            const idx = stdout.indexOf(s);
            return idx === -1 ? Infinity : idx;
          }),
        );
        if (!Number.isFinite(start)) throw new Error("bridge printed no JSON");
        const records = JSON.parse(stdout.slice(start)) as BridgeRecord[];
        return recordsToResults(records, files);
      } catch (err) {
        lastErr = err;
        // retry the next interpreter only when this one was missing at all
        // (ENOENT); a model failure or timeout must not re-run inference
        if ((err as NodeJS.ErrnoException)?.code !== "ENOENT") throw err;
      }
    }
    throw lastErr ?? new Error("no python interpreter found");
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

/** Deterministic simulated verdict (demo mode — clearly labeled in the UI). */
export function simulateFile(f: UploadFile): AnalyzeResult {
  const seed = hashBytes(new Uint8Array(f.buf));
  let s = seed >>> 0;
  const rng = () => {
    // xorshift32 — cheap and deterministic
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
  const isAI = rng() < 0.45;
  const prob = isAI ? 0.8 + 0.199 * rng() : 0.02 + 0.17 * rng();
  return {
    name: f.name,
    prob_ai: Math.round(prob * 1e6) / 1e6,
    label: prob >= 0.5 ? "AI" : "REAL",
    grid: syntheticGrid(seed, prob),
    ...imageDims(f.buf),
    bytes: f.buf.length,
    type: f.type,
  };
}

/** Python eval dumps write `NaN`; JSON.parse rejects that. */
export function parseEvalJson(text: string): unknown {
  return JSON.parse(text.replace(/\bNaN\b/g, "null"));
}

function resolveErrorFile(root: string, file: string): string | null {
  const rel = file.replace(/^[/\\]+/, "");
  const candidates = [
    path.isAbsolute(file) ? file : path.join(root, rel),
    path.join(root, rel.replace(/^runs[/\\]seer_vitl[/\\]eval_step33500/, path.join("eval", "eval_step33500"))),
    path.join(root, "client", "public", "errors", path.basename(file)),
    path.join(root, "eval", "eval_step33500", "errors_dalle_advanced", path.basename(file)),
    path.join(root, "eval", "eval_step33500", "errors_gallery", path.basename(file)),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

/** Public-folder assets become `/errors/…`; everything else stays repo-relative for /api/eval-image. */
function toClientErrorFile(root: string, abs: string): string {
  const publicDir = path.join(root, "client", "public");
  const relPublic = path.relative(publicDir, abs);
  if (!relPublic.startsWith("..") && !path.isAbsolute(relPublic)) {
    return toPosix(relPublic);
  }
  return toPosix(path.relative(root, abs));
}

function suiteName(r: Record<string, unknown>, file: string): string {
  if (typeof r.suite_name === "string" && r.suite_name) return r.suite_name;
  const base = path.basename(file, ".json");
  if (base && base !== "folders" && base !== "summary") return base;
  if (typeof r.dataset === "string" && r.dataset !== "folders") return r.dataset;
  return base;
}

/** Normalize one raw eval JSON (seer/eval.py --out-json) for the client. */
export function normalizeEvalJson(
  raw: unknown,
  file: string,
  root: string,
): EvalDataset | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const metrics = r.macro_accuracy as number | undefined;
  const sweep =
    r.perturbation_sweep && typeof r.perturbation_sweep === "object"
      ? (r.perturbation_sweep as Record<string, MetricsRow>)
      : undefined;
  const hasBody =
    typeof metrics === "number" ||
    typeof r.n === "number" ||
    !!sweep ||
    !!objOrUndef(r.per_architecture) ||
    !!objOrUndef(r.per_distorted) ||
    Array.isArray(r.error_analysis);
  if (!hasBody) return null;

  let errors: ErrorEntry[] | undefined;
  if (Array.isArray(r.error_analysis)) {
    errors = (r.error_analysis as Array<Record<string, unknown>>)
      .map((e) => {
        const listed = typeof e.file === "string" ? e.file : null;
        const abs = listed ? resolveErrorFile(root, listed) : null;
        const generator =
          typeof e.generator === "string" && /^[0-9a-f]{20,}$/i.test(e.generator)
            ? "DALL·E 3 Advanced"
            : typeof e.generator === "string"
              ? e.generator
              : undefined;
        return {
          kind: e.kind === "fp" ? ("fp" as const) : ("fn" as const),
          rank: Number(e.rank ?? 0),
          file: abs ? toClientErrorFile(root, abs) : undefined,
          imageAvailable: !!abs,
          prob_ai: Number(e.prob_ai ?? 0),
          label: (e.label === 1 ? 1 : 0) as 0 | 1,
          explained: Boolean(e.explained),
          generator,
          distortions: Array.isArray(e.distortions) ? (e.distortions as string[]) : [],
        };
      })
      .filter((e) => e.kind === "fp" || e.kind === "fn");
    if (!errors.length) errors = undefined;
  }

  const base = (sweep?.clean ?? (hasBody ? r : undefined)) as MetricsRow | undefined;
  return {
    id: file,
    name: suiteName(r, file),
    file: toPosix(path.relative(root, file)),
    checkpoint: typeof r.checkpoint === "string" ? r.checkpoint : undefined,
    step: numOrUndef(r.step),
    perturbation: typeof r.perturbation === "string" ? r.perturbation : "clean",
    metrics: (base ?? ({} as MetricsRow)) as MetricsRow,
    sweep,
    per_architecture: objOrUndef(r.per_architecture) as Record<string, MetricsRow> | undefined,
    per_distorted: objOrUndef(r.per_distorted) as Record<string, MetricsRow> | undefined,
    per_distortion: objOrUndef(r.per_distortion) as Record<string, MetricsRow> | undefined,
    robust_auroc: numOrUndef(r.robust_auroc),
    robust_mAP: numOrUndef(r.robust_mAP),
    robust_n: numOrUndef(r.robust_n),
    errors,
  };
}

function objOrUndef(v: unknown): Record<string, unknown> | undefined {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : undefined;
}
function numOrUndef(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}
function toPosix(p: string): string {
  return p.split(path.sep).join("/");
}

/** Scan eval/eval_step33500, then runs/eval and runs, newest first. */
export function scanEvalRuns(root: string, limit = 30): EvalDataset[] {
  const out: EvalDataset[] = [];
  const seen = new Set<string>();
  const dirs = [
    path.join(root, "eval", "eval_step33500"),
    path.join(root, "runs", "eval"),
    path.join(root, "runs"),
  ];
  for (const dir of dirs) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    const jsons = entries
      .filter((e) => e.isFile() && e.name.endsWith(".json") && e.name !== "summary.json")
      .map((e) => {
        const p = path.join(dir, e.name);
        return { p, mtime: fs.statSync(p).mtimeMs };
      })
      .sort((a, b) => b.mtime - a.mtime);
    for (const { p } of jsons) {
      if (seen.has(p) || out.length >= limit) continue;
      seen.add(p);
      try {
        const st = fs.statSync(p);
        if (st.size > 20 * 1024 * 1024) continue;
        const parsed = parseEvalJson(fs.readFileSync(p, "utf-8"));
        const ds = normalizeEvalJson(parsed, p, root);
        if (ds) out.push(ds);
      } catch {
        /* unreadable / not an eval file — skip */
      }
    }
  }
  return out;
}

/** Validate an /api/eval-image src param: must resolve inside <root>/runs or <root>/eval. */
export function safeEvalImagePath(root: string, rel: string): string | null {
  const abs = path.resolve(root, rel);
  const norm = path.normalize(abs);
  const allowed = [path.resolve(root, "runs"), path.resolve(root, "eval")];
  if (!allowed.some((base) => norm.startsWith(base + path.sep))) return null;
  if (!/\.(png|jpe?g|webp)$/i.test(norm)) return null;
  return fs.existsSync(norm) ? norm : null;
}
