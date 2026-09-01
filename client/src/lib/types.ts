/** Shared types across the Seer dashboard (client + API). */

/** One per-patch probability grid (G x G, e.g. 32x32 at res 512 / patch 16). */
export type PatchGrid = number[][];

/** Metrics row exactly as `seer/eval.py` writes it (all rates in 0..1). */
export interface MetricsRow {
  n?: number;
  n_fake?: number;
  n_real?: number;
  accuracy?: number;
  macro_accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  fpr?: number;
  fnr?: number;
  auroc?: number;
  mAP?: number;
  [key: string]: unknown;
}

/** One error-analysis record (`--error-dir` output). */
export interface ErrorEntry {
  kind: "fp" | "fn";
  rank: number;
  /** `errors/<name>` for files in client/public, else a repo-relative path for /api/eval-image */
  file?: string;
  /** true when `file` exists (static public asset or /api/eval-image) */
  imageAvailable?: boolean;
  prob_ai: number;
  /** ground truth: 1 = AI, 0 = real */
  label: 0 | 1;
  explained: boolean;
  generator?: string;
  distortions?: string[];
  /** demo mode only: synthetic heatmap grid so the panel still renders */
  grid?: PatchGrid;
}

/** A normalized eval run (from runs/eval/*.json, or bundled demo data). */
export interface EvalDataset {
  id: string;
  /** dataset key, e.g. "ntire_val" */
  name: string;
  /** file the metrics came from (live mode) */
  file?: string;
  checkpoint?: string;
  step?: number;
  perturbation?: string;
  metrics: MetricsRow;
  sweep?: Record<string, MetricsRow>;
  per_architecture?: Record<string, MetricsRow>;
  per_distorted?: Record<string, MetricsRow>;
  per_distortion?: Record<string, MetricsRow>;
  robust_auroc?: number;
  robust_mAP?: number;
  robust_n?: number;
  errors?: ErrorEntry[];
  demo?: boolean;
}

export interface EvalResponse {
  mode: "live" | "demo";
  root?: string | null;
  datasets: EvalDataset[];
  note?: string;
}

/** Result of POST /api/analyze. */
export interface AnalyzeResult {
  name: string;
  prob_ai: number;
  label: "AI" | "REAL";
  grid: PatchGrid | null;
  width?: number;
  height?: number;
  bytes?: number;
  type?: string;
  /** ms spent in the model (live mode) */
  elapsedMs?: number;
}

/** Modal deployment status, probed by /api/status when SEER_MODAL_URL is set. */
export interface ModalStatus {
  url: string;
  /** true = healthy, false = answered but not ready, null = unreachable (cold or down) */
  ok: boolean | null;
  device?: string | null;
  checkpoint?: string | null;
  error?: string | null;
}

/** Result of POST /api/analyze. There is no simulated mode — either a
 * backend scores the images or the request fails with an error. */
export interface AnalyzeResponse {
  backend: "local" | "modal";
  checkpoint?: string | null;
  results: AnalyzeResult[];
}

export interface StatusResponse {
  mode: "live" | "unavailable";
  checkpoint: string | null;
  root: string | null;
  uv: string | null;
  /** persistent seer_serve.py URL when it answered /health */
  server?: string | null;
  device?: string | null;
  error?: string | null;
  /** Modal deployment (client/scripts/modal_seer.py) when SEER_MODAL_URL is set */
  modal?: ModalStatus | null;
}
