/**
 * Bundled demo results, used when no real eval JSONs exist under `runs/`.
 * Shapes mirror `seer/eval.py --out-json` exactly so the UI code path is
 * identical between demo and live. Numbers are plausible placeholders for the
 * hero recipe — every dataset is tagged `demo: true` and clearly labeled.
 */
import { syntheticGrid } from "./heat";
import type { ErrorEntry, EvalDataset, MetricsRow } from "./types";

const row = (m: Partial<MetricsRow>): MetricsRow => m as MetricsRow;

/** NTIRE 2026 val — full benchmark perturbation sweep (16 passes). */
const ntireSweep: Record<string, MetricsRow> = {
  clean: row({ macro_accuracy: 0.9908, mAP: 0.9971, auroc: 0.9994, f1: 0.9905, fpr: 0.012, fnr: 0.007, n: 12000, n_fake: 6000, n_real: 6000 }),
  jpeg90: row({ macro_accuracy: 0.9894, mAP: 0.9963, auroc: 0.9991, f1: 0.9891, fpr: 0.014, fnr: 0.008, n: 12000 }),
  jpeg70: row({ macro_accuracy: 0.9861, mAP: 0.9947, auroc: 0.9986, f1: 0.9858, fpr: 0.019, fnr: 0.011, n: 12000 }),
  jpeg50: row({ macro_accuracy: 0.9802, mAP: 0.9918, auroc: 0.9979, f1: 0.9799, fpr: 0.026, fnr: 0.015, n: 12000 }),
  jpeg30: row({ macro_accuracy: 0.9644, mAP: 0.9841, auroc: 0.9952, f1: 0.9639, fpr: 0.041, fnr: 0.027, n: 12000 }),
  "blur0.5": row({ macro_accuracy: 0.9885, mAP: 0.9956, auroc: 0.9989, f1: 0.9882, fpr: 0.016, fnr: 0.009, n: 12000 }),
  "blur1.0": row({ macro_accuracy: 0.9812, mAP: 0.9922, auroc: 0.9977, f1: 0.9809, fpr: 0.023, fnr: 0.014, n: 12000 }),
  "blur2.0": row({ macro_accuracy: 0.9630, mAP: 0.9835, auroc: 0.9948, f1: 0.9625, fpr: 0.044, fnr: 0.030, n: 12000 }),
  "resize0.5": row({ macro_accuracy: 0.9877, mAP: 0.9951, auroc: 0.9988, f1: 0.9874, fpr: 0.017, fnr: 0.010, n: 12000 }),
  "resize0.25": row({ macro_accuracy: 0.9741, mAP: 0.9889, auroc: 0.9966, f1: 0.9737, fpr: 0.031, fnr: 0.019, n: 12000 }),
  "noise0.02": row({ macro_accuracy: 0.9890, mAP: 0.9959, auroc: 0.9990, f1: 0.9887, fpr: 0.015, fnr: 0.008, n: 12000 }),
  "noise0.05": row({ macro_accuracy: 0.9831, mAP: 0.9931, auroc: 0.9981, f1: 0.9828, fpr: 0.021, fnr: 0.012, n: 12000 }),
  "noise0.10": row({ macro_accuracy: 0.9688, mAP: 0.9857, auroc: 0.9959, f1: 0.9683, fpr: 0.037, fnr: 0.024, n: 12000 }),
  jitter20: row({ macro_accuracy: 0.9896, mAP: 0.9961, auroc: 0.9990, f1: 0.9893, fpr: 0.014, fnr: 0.009, n: 12000 }),
  crop80: row({ macro_accuracy: 0.9901, mAP: 0.9966, auroc: 0.9992, f1: 0.9898, fpr: 0.013, fnr: 0.008, n: 12000 }),
  pangram: row({ macro_accuracy: 0.9865, mAP: 0.9942, auroc: 0.9984, f1: 0.9862, fpr: 0.018, fnr: 0.012, n: 12000 }),
};

/** Most confident mistakes (demo): FPs are digital-art-style reals, FNs are
 *  heavily compressed generated images — the failure modes the README calls out. */
const demoErrors: ErrorEntry[] = [
  {
    kind: "fp", rank: 1, prob_ai: 0.994, label: 0, explained: true,
    generator: "wikiart-style digital painting", distortions: [],
    grid: syntheticGrid(101, 0.994),
  },
  {
    kind: "fp", rank: 2, prob_ai: 0.971, label: 0, explained: true,
    generator: "high-ISO night photography", distortions: ["noise"],
    grid: syntheticGrid(102, 0.971),
  },
  {
    kind: "fp", rank: 3, prob_ai: 0.958, label: 0, explained: true,
    generator: "watercolor illustration", distortions: [],
    grid: syntheticGrid(103, 0.958),
  },
  {
    kind: "fp", rank: 4, prob_ai: 0.912, label: 0, explained: true,
    generator: "bokeh portrait (f/1.4)", distortions: ["blur"],
    grid: syntheticGrid(104, 0.912),
  },
  {
    kind: "fn", rank: 1, prob_ai: 0.006, label: 1, explained: true,
    generator: "flux.1-dev", distortions: ["jpeg"],
    grid: syntheticGrid(201, 0.006),
  },
  {
    kind: "fn", rank: 2, prob_ai: 0.041, label: 1, explained: true,
    generator: "sdxl-turbo", distortions: ["blur"],
    grid: syntheticGrid(202, 0.041),
  },
  {
    kind: "fn", rank: 3, prob_ai: 0.187, label: 1, explained: true,
    generator: "midjourney-6", distortions: ["noise"],
    grid: syntheticGrid(203, 0.187),
  },
  {
    kind: "fn", rank: 4, prob_ai: 0.334, label: 1, explained: true,
    generator: "ideogram-3.0", distortions: ["resize"],
    grid: syntheticGrid(204, 0.334),
  },
];

export const DEMO_DATASETS: EvalDataset[] = [
  {
    id: "demo-ntire-val",
    name: "ntire_val",
    demo: true,
    checkpoint: "runs/seer_vitl/best.pt (demo)",
    perturbation: "clean",
    metrics: ntireSweep.clean,
    sweep: ntireSweep,
    per_distorted: {
      clean: row({ macro_accuracy: 0.9921, auroc: 0.9995, f1: 0.9918, n: 6400 }),
      distorted: row({ macro_accuracy: 0.9702, auroc: 0.9941, f1: 0.9697, n: 5600 }),
    },
    per_distortion: {
      jpeg: row({ macro_accuracy: 0.9695, f1: 0.9690, n: 2140 }),
      webp: row({ macro_accuracy: 0.9731, f1: 0.9726, n: 860 }),
      blur: row({ macro_accuracy: 0.9782, f1: 0.9778, n: 740 }),
      noise: row({ macro_accuracy: 0.9668, f1: 0.9662, n: 690 }),
      resize: row({ macro_accuracy: 0.9815, f1: 0.9811, n: 620 }),
      combined: row({ macro_accuracy: 0.9489, f1: 0.9482, n: 550 }),
    },
    robust_auroc: 0.9941,
    robust_mAP: 0.9872,
    robust_n: 5600,
    errors: demoErrors,
  },
  {
    id: "demo-comfor-eval",
    name: "comfor_eval",
    demo: true,
    checkpoint: "runs/seer_vitl/best.pt (demo)",
    perturbation: "clean",
    metrics: row({
      macro_accuracy: 0.9714, mAP: 0.9938, auroc: 0.9974, f1: 0.9719,
      fpr: 0.031, fnr: 0.025, n: 51800, n_fake: 24100, n_real: 27700,
    }),
    per_architecture: {
      "sd15": row({ macro_accuracy: 0.9962, f1: 0.9962, auroc: 0.9998, n: 5200 }),
      "sdxl": row({ macro_accuracy: 0.9918, f1: 0.9917, auroc: 0.9996, n: 5400 }),
      "flux.1-dev": row({ macro_accuracy: 0.9785, f1: 0.9782, auroc: 0.9987, n: 3800 }),
      "midjourney-6": row({ macro_accuracy: 0.9621, f1: 0.9614, auroc: 0.9962, n: 3600 }),
      "dalle3": row({ macro_accuracy: 0.9552, f1: 0.9543, auroc: 0.9951, n: 3100 }),
      "ideogram-3.0": row({ macro_accuracy: 0.9340, f1: 0.9326, auroc: 0.9918, n: 2900 }),
      "photos (real)": row({ macro_accuracy: 0.9780, f1: 0.9790, auroc: 0.9971, n: 27700 }),
    },
  },
];
