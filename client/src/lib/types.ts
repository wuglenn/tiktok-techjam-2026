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
  /** absolute path on disk (live mode) */
  file?: string;
  /** true when `file` exists and can be served by /api/eval-image */
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

export interface AnalyzeResponse {
  mode: "live" | "simulated";
  checkpoint?: string | null;
  note?: string | null;
  results: AnalyzeResult[];
}

export interface StatusResponse {
  mode: "live" | "simulated";
  checkpoint: string | null;
  root: string | null;
  uv: string | null;
  error?: string | null;
}
