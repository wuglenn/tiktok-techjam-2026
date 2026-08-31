/** Small formatting helpers (all metrics are 0..1 rates). */

/** 0.9871 -> "98.71" */
export function pct(v: number | undefined | null, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  return `${(v * 100).toFixed(digits)}`;
}

/** 0.9871 -> "98.71%" */
export function pctS(v: number | undefined | null, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  return `${(v * 100).toFixed(digits)}%`;
}

/** 0.0034 -> "0.34pp" (percentage points, relative to a 0..1 rate) */
export function pp(v: number | undefined | null, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  const s = (v * 100).toFixed(digits);
  return `${v > 0 ? "+" : ""}${s}pp`;
}

/** 1283123 -> "1.3M" */
export function compact(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return "–";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

/** 153600 -> "150 KB" */
export function bytes(n: number | undefined | null): string {
  if (n == null || Number.isNaN(n)) return "–";
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

/** Pretty labels for the perturbation keys used by seer/augment.py. */
export const PERTURBATION_LABELS: Record<string, string> = {
  clean: "Clean (no perturbation)",
  jpeg90: "JPEG q90",
  jpeg70: "JPEG q70",
  jpeg50: "JPEG q50",
  jpeg30: "JPEG q30",
  "blur0.5": "Blur σ 0.5",
  "blur1.0": "Blur σ 1.0",
  "blur2.0": "Blur σ 2.0",
  "resize0.5": "Resize 0.5×",
  "resize0.25": "Resize 0.25×",
  "noise0.02": "Noise σ 0.02",
  "noise0.05": "Noise σ 0.05",
  "noise0.10": "Noise σ 0.10",
  jitter20: "Jitter ±20%",
  crop80: "Center crop 80%",
  pangram: "Pangram protocol (1024px + JPEG q50)",
};

/** Family grouping for the robustness table. */
export const PERTURBATION_FAMILIES: { name: string; keys: string[] }[] = [
  { name: "Compression", keys: ["jpeg90", "jpeg70", "jpeg50", "jpeg30"] },
  { name: "Blur", keys: ["blur0.5", "blur1.0", "blur2.0"] },
  { name: "Rescale", keys: ["resize0.5", "resize0.25"] },
  { name: "Noise", keys: ["noise0.02", "noise0.05", "noise0.10"] },
  { name: "Color", keys: ["jitter20"] },
  { name: "Geometry", keys: ["crop80"] },
  { name: "Protocol", keys: ["pangram"] },
];

export function perturbationLabel(key: string): string {
  return PERTURBATION_LABELS[key] ?? key;
}

export function familyOf(key: string): string {
  for (const f of PERTURBATION_FAMILIES) {
    if (f.keys.includes(key)) return f.name;
  }
  return "Other";
}
