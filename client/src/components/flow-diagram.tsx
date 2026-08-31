import { syntheticGrid, turbo, turboGradientCss } from "@/lib/heat";

export function FlowDiagram() {
  return (
    <div className="mx-auto max-w-md">
      <div className="border border-dashed border-rule px-4 py-3 text-center">
        <span className="chip">input</span>
        <div className="small-head mt-2">image · any size</div>
        <div className="caption mt-1">
          upscaled to 512px when smaller — every image gets a verdict
        </div>
      </div>

      <VArrow label="(3, 512, 512) · ImageNet norm" />

      <div className="border border-dashed border-rule bg-paper-wash px-4 py-3.5 text-center">
        <span className="chip">backbone</span>
        <div className="small-head mt-2">DINOv3 ViT-L/16</div>
        <div className="caption mt-1">
          24 transformer blocks · ~300M params · full continuation fine-tuning
          (layer-wise LR decay 0.8)
        </div>
      </div>

      <Fork />

      <div className="grid grid-cols-2 gap-3 sm:gap-6">
        <Branch label="CLS token">
          <div className="border border-dashed border-rule px-3.5 py-3 text-center">
            <div className="small-head text-[14px]">global head</div>
            <div className="caption mt-0.5">MLP · page-level</div>
          </div>
          <VArrow label="sigmoid" compact />
          <div className="flex flex-1 flex-col justify-center border border-dashed border-rule px-3.5 py-3 text-center">
            <div className="small-head text-[14px]">P(AI)</div>
            <div className="mt-1.5 flex items-center gap-2">
              <span className="metric-bar flex-1">
                <span style={{ width: "98.5%" }} />
              </span>
              <span className="tabular font-mono text-[12px] text-ink-head">
                0.987
              </span>
            </div>
            <div className="caption mt-1">one logit per image</div>
          </div>
        </Branch>

        <Branch label="patch tokens ×1024">
          <div className="border border-dashed border-rule px-3.5 py-3 text-center">
            <div className="small-head text-[14px]">local head</div>
            <div className="caption mt-0.5">linear · per patch</div>
          </div>
          <VArrow label="sigmoid ×1024" compact />
          <div className="border border-dashed border-rule px-3.5 py-3 text-center">
            <div className="small-head text-[14px]">32×32 probability grid</div>
            <PatchGrid prob={0.987} />
            <div className="mt-2 flex items-center gap-2">
              <span className="tabular font-mono text-[11px] text-ink-mute">0</span>
              <span
                className="h-1 flex-1 rounded-sm"
                style={{ background: turboGradientCss() }}
              />
              <span className="tabular font-mono text-[11px] text-ink-mute">1</span>
            </div>
            <div className="caption mt-1.5">
              one probability per patch — warm cells mark generated regions
            </div>
          </div>
        </Branch>
      </div>

      <Converge />

      <div className="border border-dashed border-rule px-4 py-3.5">
        <div className="flex items-center justify-center">
          <span className="chip">output</span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div className="border border-dashed border-rule px-3 py-2 text-center">
            <div className="small-head text-[14px]">verdict</div>
            <div className="caption mt-0.5">P(AI) ≥ 0.5 → AI</div>
          </div>
          <div className="border border-dashed border-rule px-3 py-2 text-center">
            <div className="small-head text-[14px]">heatmap</div>
            <div className="caption mt-0.5">grid upsampled · turbo overlay</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PatchGrid({ prob, seed = 11 }: { prob: number; seed?: number }) {
  const grid = syntheticGrid(seed, prob);
  return (
    <div
      className="figure-frame mx-auto mt-2 grid aspect-square w-full"
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
      <span className="absolute inset-y-1 left-1/2 w-px -translate-x-1/2 bg-rule" />
      {label && (
        <span className="relative z-10 max-w-[85%] truncate bg-paper-wash px-2 font-mono text-[12px] text-ink-mute">
          {label}
        </span>
      )}
    </div>
  );
}

function Fork() {
  return (
    <svg
      viewBox="0 0 100 40"
      preserveAspectRatio="none"
      className="h-10 w-full text-rule"
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

function Converge() {
  return (
    <svg
      viewBox="0 0 100 40"
      preserveAspectRatio="none"
      className="h-10 w-full text-rule"
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
    </svg>
  );
}

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
        <span className="max-w-full truncate font-mono text-[12px] text-ink-mute">
          {label}
        </span>
      </div>
      <div className="flex flex-1 flex-col">{children}</div>
    </div>
  );
}
