import { syntheticGrid, turbo, turboGradientCss } from "@/lib/heat";

/**
 * Data-flow diagram for the architecture section: input -> backbone -> fork
 * into global/local heads -> verdict + heatmap. Pure CSS + SVG connectors,
 * no diagram dependency, rendered on the server.
 */
export function FlowDiagram() {
  return (
    <div className="panel mx-auto max-w-2xl p-5 sm:p-7">
      <div className="mx-auto max-w-md">
        {/* --------------------------------------------------------- input */}
        <div className="rounded-xl border border-dashed border-white/[0.15] bg-white/[0.02] px-4 py-3 text-center">
          <Badge>input</Badge>
          <div className="mt-1.5 text-sm font-semibold text-white">
            image · any size
          </div>
          <div className="mt-0.5 text-[11px] text-zinc-500">
            upscaled to 512px when smaller — every image gets a verdict
          </div>
        </div>

        <VArrow label="(3, 512, 512) · ImageNet norm" />

        {/* ----------------------------------------------------- backbone */}
        <div className="rounded-xl border border-cyan-400/25 bg-cyan-500/[0.06] px-4 py-3.5 text-center shadow-[0_0_36px_-12px_rgba(34,211,238,0.35)]">
          <Badge tone="cyan">backbone</Badge>
          <div className="mt-1.5 text-sm font-semibold text-white">
            DINOv3 ViT-L/16
          </div>
          <div className="mt-0.5 text-[11px] leading-relaxed text-zinc-400">
            24 transformer blocks · ~300M params · full continuation
            fine-tuning (layer-wise LR decay 0.8)
          </div>
        </div>

        <Fork />

        {/* -------------------------------------------------- dual heads */}
        <div className="grid grid-cols-2 gap-3 sm:gap-6">
          <Branch label="CLS token">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-3.5 py-3 text-center">
              <div className="text-xs font-semibold text-white">global head</div>
              <div className="mt-0.5 text-[11px] text-zinc-500">
                MLP · page-level
              </div>
            </div>
            <VArrow label="sigmoid" compact />
            <div className="flex flex-1 flex-col justify-center rounded-xl border border-white/[0.08] bg-white/[0.03] px-3.5 py-3 text-center">
              <div className="text-xs font-semibold text-white">P(AI)</div>
              <div className="mt-1.5 flex items-center gap-2">
                <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                  <span
                    className="absolute inset-0 rounded-full bg-[linear-gradient(90deg,#34d399_0%,#fbbf24_50%,#fb7185_100%)]"
                    style={{ clipPath: "inset(0 1.5% 0 0)" }}
                  />
                </span>
                <span className="tabular text-[11px] font-semibold text-rose-300">
                  0.987
                </span>
              </div>
              <div className="mt-1 text-[10px] text-zinc-600">
                one logit per image
              </div>
            </div>
          </Branch>

          <Branch label="patch tokens ×1024">
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-3.5 py-3 text-center">
              <div className="text-xs font-semibold text-white">local head</div>
              <div className="mt-0.5 text-[11px] text-zinc-500">
                linear · per patch
              </div>
            </div>
            <VArrow label="sigmoid ×1024" compact />
            <div className="rounded-xl border border-white/[0.08] bg-white/[0.03] px-3.5 py-3 text-center">
              <div className="text-xs font-semibold text-white">
                32×32 probability grid
              </div>
              <PatchGrid prob={0.987} />
              <div className="mt-2 flex items-center gap-2">
                <span className="tabular text-[9px] text-zinc-600">0</span>
                <span
                  className="h-1 flex-1 rounded-full ring-1 ring-white/10"
                  style={{ background: turboGradientCss() }}
                />
                <span className="tabular text-[9px] text-zinc-600">1</span>
              </div>
              <div className="mt-1.5 text-[10px] text-zinc-600">
                one probability per patch — warm cells mark generated regions
              </div>
            </div>
          </Branch>
        </div>

        <Converge />

        {/* ------------------------------------------------------- output */}
        <div className="rounded-xl border border-white/[0.12] bg-white/[0.04] px-4 py-3.5">
          <div className="flex items-center justify-center">
            <Badge>output</Badge>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-rose-400/20 bg-rose-500/[0.06] px-3 py-2 text-center">
              <div className="text-[11px] font-semibold text-rose-300">
                verdict
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-500">
                P(AI) ≥ 0.5 → AI
              </div>
            </div>
            <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/[0.06] px-3 py-2 text-center">
              <div className="text-[11px] font-semibold text-cyan-300">
                heatmap
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-500">
                grid upsampled · turbo overlay
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- pieces */

/**
 * The local head's output rendered as the real 32×32 grid: one cell per
 * patch, turbo-colored by its sigmoid probability. Deterministic
 * (seeded blob field), so it renders identically on the server.
 */
function PatchGrid({ prob, seed = 11 }: { prob: number; seed?: number }) {
  const grid = syntheticGrid(seed, prob);
  return (
    <div
      className="mx-auto mt-2 grid aspect-square w-full overflow-hidden rounded-md ring-1 ring-white/10"
      style={{
        gridTemplateColumns: "repeat(32, minmax(0, 1fr))",
        gridTemplateRows: "repeat(32, minmax(0, 1fr))",
      }}
      role="img"
      aria-label="32 by 32 patch probability grid"
    >
      {grid.map((row, y) =>
        row.map((v, x) => {
          const [r, g, b] = turbo(v);
          return (
            <div
              key={`${y}-${x}`}
              style={{ backgroundColor: `rgb(${r},${g},${b})` }}
            />
          );
        }),
      )}
    </div>
  );
}

function Badge({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone?: "cyan";
}) {
  return (
    <span
      className={`inline-block rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.16em] ${
        tone === "cyan"
          ? "bg-cyan-500/15 text-cyan-300"
          : "bg-white/[0.06] text-zinc-500"
      }`}
    >
      {children}
    </span>
  );
}

/** Vertical connector with an optional label chip masking the line. */
function VArrow({
  label,
  compact = false,
}: {
  label?: string;
  compact?: boolean;
}) {
  return (
    <div
      className={`relative flex items-center justify-center ${compact ? "h-9" : "h-11"}`}
    >
      <span className="absolute inset-y-1 left-1/2 w-px -translate-x-1/2 bg-linear-to-b from-white/[0.18] to-white/[0.28]" />
      <span className="absolute bottom-0 left-1/2 h-0 w-0 -translate-x-1/2 border-x-[3.5px] border-x-transparent border-t-[4.5px] border-t-white/30" />
      {label && (
        <span className="relative z-10 max-w-[85%] truncate rounded-full border border-white/[0.07] bg-[#0c0c0f] px-2.5 py-0.5 text-[10px] text-zinc-400">
          {label}
        </span>
      )}
    </div>
  );
}

/** Fork connector: one line in at center top, two out at the column centers. */
function Fork() {
  return (
    <svg
      viewBox="0 0 100 40"
      preserveAspectRatio="none"
      className="h-10 w-full text-white/[0.22]"
      aria-hidden
    >
      <path
        d="M50,1 C50,20 25,16 25,39"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
      <path
        d="M50,1 C50,20 75,16 75,39"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/** Converge connector: two lines in from the column centers, one out. */
function Converge() {
  return (
    <svg
      viewBox="0 0 100 40"
      preserveAspectRatio="none"
      className="h-10 w-full text-white/[0.22]"
      aria-hidden
    >
      <path
        d="M25,1 C25,20 50,20 50,33"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
      <path
        d="M75,1 C75,20 50,20 50,33"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
      <path
        d="M50,33 L50,39"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
      <path
        d="M46.5,36 L50,40 L53.5,36"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/** One head branch: fork label chip + its nodes. */
function Branch({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-w-0 flex-col">
      <div className="mb-2 flex justify-center">
        <span className="max-w-full truncate rounded-full border border-white/[0.07] bg-[#0c0c0f] px-2.5 py-0.5 text-[10px] text-zinc-400">
          {label}
        </span>
      </div>
      <div className="flex flex-1 flex-col">{children}</div>
    </div>
  );
}
