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
 * Best checkpoint to serve: $SEER_CHECKPOINT, else the newest
 * `runs/<name>/best.pt` (preferring the hero `seer_vitl*` runs).
 */
export function findCheckpoint(root: string): string | null {
  const env = process.env.SEER_CHECKPOINT;
  if (env) {
    const p = path.isAbsolute(env) ? env : path.join(root, env);
    if (fs.existsSync(p)) return p;
  }
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

/**
 * Live inference: save uploads to a scratch dir, run the Python bridge,
 * return one record per image. Throws with a readable message on failure.
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
        const records = JSON.parse(stdout.slice(start)) as Array<{
          image: string;
          prob_ai: number;
          label: string;
          grid: number[][] | null;
          width?: number;
          height?: number;
        }>;
        return records.map((r, i) => ({
          name: files[i]?.name ?? path.basename(r.image),
          prob_ai: r.prob_ai,
          label: r.label === "AI" ? ("AI" as const) : ("REAL" as const),
          grid: r.grid,
          width: r.width,
          height: r.height,
          bytes: files[i]?.buf.length,
          type: files[i]?.type,
        }));
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
  if (typeof metrics !== "number" && !sweep) return null;

  let errors: ErrorEntry[] | undefined;
  if (Array.isArray(r.error_analysis)) {
    errors = (r.error_analysis as Array<Record<string, unknown>>)
      .map((e) => {
        const abs = typeof e.file === "string" ? e.file : null;
        const available = !!abs && fs.existsSync(abs);
        return {
          kind: e.kind === "fp" ? ("fp" as const) : ("fn" as const),
          rank: Number(e.rank ?? 0),
          file: available && abs ? toPosix(path.relative(root, abs)) : undefined,
          imageAvailable: available,
          prob_ai: Number(e.prob_ai ?? 0),
          label: (e.label === 1 ? 1 : 0) as 0 | 1,
          explained: Boolean(e.explained),
          generator: typeof e.generator === "string" ? e.generator : undefined,
          distortions: Array.isArray(e.distortions) ? (e.distortions as string[]) : [],
        };
      })
      .filter((e) => e.kind === "fp" || e.kind === "fn");
    if (!errors.length) errors = undefined;
  }

  const base = (sweep?.clean ?? (typeof metrics === "number" ? r : undefined)) as
    | MetricsRow
    | undefined;
  return {
    id: file,
    name: typeof r.dataset === "string" ? r.dataset : path.basename(file, ".json"),
    file: toPosix(path.relative(root, file)),
    checkpoint: typeof r.checkpoint === "string" ? r.checkpoint : undefined,
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
  return typeof v === "number" ? v : undefined;
}
function toPosix(p: string): string {
  return p.split(path.sep).join("/");
}

/** Scan runs/eval/*.json and runs/*.json for eval metrics (newest first). */
export function scanEvalRuns(root: string, limit = 30): EvalDataset[] {
  const out: EvalDataset[] = [];
  const seen = new Set<string>();
  for (const sub of ["runs", path.join("runs", "eval")]) {
    const dir = path.join(root, sub);
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    const jsons = entries
      .filter((e) => e.isFile() && e.name.endsWith(".json"))
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
        const parsed = JSON.parse(fs.readFileSync(p, "utf-8"));
        const ds = normalizeEvalJson(parsed, p, root);
        if (ds) out.push(ds);
      } catch {
        /* unreadable / not an eval file — skip */
      }
    }
  }
  return out;
}

/** Validate an /api/eval-image src param: must resolve inside <root>/runs. */
export function safeEvalImagePath(root: string, rel: string): string | null {
  const abs = path.resolve(root, rel);
  const runsRoot = path.resolve(root, "runs");
  const norm = path.normalize(abs);
  if (!norm.startsWith(runsRoot + path.sep)) return null;
  if (!/\.(png|jpe?g|webp)$/i.test(norm)) return null;
  return fs.existsSync(norm) ? norm : null;
}
