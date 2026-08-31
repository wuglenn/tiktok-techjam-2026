import { evalKey, finite } from "@/lib/eval-labels";
import type { EvalDataset } from "@/lib/types";

/** Published NTIRE 2026 open-test leaderboard, ranked by robust ROC AUC. */
export interface NtireEntry {
  method: string;
  auroc: number;
  robust: number;
  params: string;
  ours?: boolean;
}

export const NTIRE_OPEN_TEST: NtireEntry[] = [
  { method: "MICV", auroc: 0.9978, robust: 0.9738, params: "7B" },
  { method: "Ant International", auroc: 0.9973, robust: 0.9731, params: "7B" },
  { method: "TeleAI-TeleGuard", auroc: 0.9762, robust: 0.9215, params: "7B" },
  { method: "INTSIG", auroc: 0.981, robust: 0.909, params: "7B" },
  { method: "vincentlc", auroc: 0.9497, robust: 0.8633, params: "7B" },
  { method: "UESTC", auroc: 0.9693, robust: 0.8558, params: "7B" },
  { method: "Reagvis Labs", auroc: 0.9423, robust: 0.8474, params: "7B" },
  { method: "PSU", auroc: 0.9132, robust: 0.8334, params: "7B" },
  { method: "Shallow Real", auroc: 0.9954, robust: 0.8302, params: "7B" },
];

/** last.pt at step 33,500 on the public test — used if live eval is missing. */
export const SEER_NTIRE_FALLBACK: NtireEntry = {
  method: "Seer",
  auroc: 0.9676961538461539,
  robust: 0.9228166666666666,
  params: "302M",
  ours: true,
};

export function seerNtireFromEval(datasets: EvalDataset[]): NtireEntry | null {
  const ds = datasets.find((d) => evalKey(d.name, d.file) === "ntire_test");
  if (!ds) return null;
  const auroc = finite(ds.metrics.auroc);
  const robust =
    finite(ds.robust_auroc) ?? finite(ds.per_distorted?.distorted?.auroc);
  if (auroc == null || robust == null) return null;
  return { method: "Seer", auroc, robust, params: "302M", ours: true };
}

export function withSeer(
  published: NtireEntry[],
  seer: NtireEntry | null,
): NtireEntry[] {
  const rows = seer ? [...published, seer] : [...published];
  return rows.sort((a, b) => b.robust - a.robust || b.auroc - a.auroc);
}

export function columnMarks(rows: NtireEntry[], key: "auroc" | "robust") {
  const uniq = [...new Set(rows.map((r) => r[key]))].sort((a, b) => b - a);
  return { best: uniq[0], second: uniq[2] };
}
