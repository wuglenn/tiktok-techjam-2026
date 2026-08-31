/**
 * Heatmap utilities shared by the server (simulation, demo data) and the
 * client (canvas rendering). Mirrors the Python side's `turbo` colormap so
 * dashboard overlays match the matplotlib panels written by `seer/heatmap.py`.
 */

export type RGB = [number, number, number];

/** Google "Turbo" colormap polynomial approximation (Anton Mikhailov). */
export function turbo(t: number): RGB {
  const x = Math.min(1, Math.max(0, t));
  const r =
    0.13572138 +
    4.6153926 * x -
    42.66032258 * x ** 2 +
    132.13108234 * x ** 3 -
    152.94239396 * x ** 4 +
    59.28637943 * x ** 5;
  const g =
    0.09140261 +
    2.19418839 * x +
    4.84296658 * x ** 2 -
    14.18503333 * x ** 3 +
    4.27729857 * x ** 4 +
    2.82956604 * x ** 5;
  const b =
    0.1066733 +
    12.64194608 * x -
    60.58204836 * x ** 2 +
    110.36276771 * x ** 3 -
    89.90310912 * x ** 4 +
    27.34824973 * x ** 5;
  const c = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return [c(r), c(g), c(b)];
}

/** CSS gradient string of the turbo colormap, for legends. */
export function turboGradientCss(stops = 12): string {
  const parts: string[] = [];
  for (let i = 0; i < stops; i++) {
    const t = i / (stops - 1);
    const [r, g, b] = turbo(t);
    parts.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(0)}%`);
  }
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

/** Deterministic 32-bit PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a 32-bit hash of a byte buffer (first N bytes are enough in practice). */
export function hashBytes(bytes: Uint8Array, limit = 65536): number {
  let h = 0x811c9dc5;
  const n = Math.min(bytes.length, limit);
  for (let i = 0; i < n; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * Smooth blob field in [0, 1] on a G x G grid: bilinear value noise over a
 * small lattice. Deterministic per seed — same seed, same field.
 */
function blobField(seed: number, G: number, lattice = 5): number[][] {
  const rng = mulberry32(seed);
  const L = lattice;
  const vals = Array.from({ length: L * L }, () => rng());
  const field: number[][] = [];
  for (let y = 0; y < G; y++) {
    const row: number[] = [];
    const fy = (y / (G - 1)) * (L - 1);
    const y0 = Math.floor(fy);
    const ty = smoothstep(0, 1, fy - y0);
    for (let x = 0; x < G; x++) {
      const fx = (x / (G - 1)) * (L - 1);
      const x0 = Math.floor(fx);
      const tx = smoothstep(0, 1, fx - x0);
      const v00 = vals[y0 * L + x0];
      const v10 = vals[y0 * L + Math.min(x0 + 1, L - 1)];
      const v01 = vals[Math.min(y0 + 1, L - 1) * L + x0];
      const v11 = vals[Math.min(y0 + 1, L - 1) * L + Math.min(x0 + 1, L - 1)];
      const top = v00 + (v10 - v00) * tx;
      const bot = v01 + (v11 - v01) * tx;
      row.push(top + (bot - top) * ty);
    }
    field.push(row);
  }
  return field;
}

/**
 * Synthetic per-patch grid for demo/simulated verdicts: a smooth blob field
 * shaped so its hot regions line up with `prob`. Visually consistent with the
 * real local head's output (most patches near the image verdict, structured
 * hot/cold regions rather than uniform noise).
 */
export function syntheticGrid(seed: number, prob: number, G = 32): number[][] {
  const field = blobField(seed ^ 0x9e3779b9, G);
  return field.map((row) =>
    row.map((n) => {
      const mask = smoothstep(0.38, 0.78, n);
      const lo = prob * 0.22;
      const v = lo + (prob - lo) * mask;
      return Math.round(Math.min(1, Math.max(0, v)) * 1000) / 1000;
    }),
  );
}
