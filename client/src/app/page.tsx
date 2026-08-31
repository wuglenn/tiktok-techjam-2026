import Link from "next/link";

import { StatCard } from "@/components/charts";
import { HeroDemo } from "@/components/hero-demo";
import { EvalResults } from "@/components/eval-results";
import { FlowDiagram } from "@/components/flow-diagram";
import { MixtureTable } from "@/components/mixture";
import {
  IconArrowRight,
  IconChart,
  IconFlame,
  IconScan,
  IconZap,
} from "@/components/icons";

export default function OverviewPage() {
  return (
    <div className="space-y-24">
      {/* ---------------------------------------------------------- hero */}
      <section className="relative">
        <div className="pointer-events-none absolute -top-24 left-1/2 -z-10 h-72 w-[42rem] -translate-x-1/2 rounded-full bg-cyan-500/[0.07] blur-3xl" />
        <div className="grid items-center gap-12 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="animate-rise">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1 text-[11px] font-medium tracking-wide text-zinc-400">
              <IconFlame className="h-3.5 w-3.5 text-cyan-400" />
              TikTok TechJam 2026 · Track 5
            </div>
            <h1 className="mt-6 text-5xl font-semibold leading-[1.05] tracking-tight text-white sm:text-6xl">
              AI images leave
              <br />
              fingerprints.
              <span className="text-gradient"> Seer reads them.</span>
            </h1>
            <p className="mt-6 max-w-xl text-base leading-relaxed text-zinc-400">
              A 302M-parameter detector — DINOv3 ViT-L fully fine-tuned with dual
              global + patch heads — trained on public data with
              wild-simulation augmentation and composite training. One forward
              pass gives an image-level verdict{" "}
              <em className="not-italic text-zinc-200">and</em> a pixel-level
              heatmap.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/analyze"
                className="group inline-flex items-center gap-2 rounded-xl bg-linear-to-r from-cyan-400 to-sky-500 px-5 py-3 text-sm font-semibold text-zinc-950 shadow-lg shadow-cyan-500/20 transition-transform hover:scale-[1.02] active:scale-[0.99]"
              >
                <IconScan className="h-4 w-4" />
                Analyze an image
                <IconArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </Link>
              <Link
                href="/robustness"
                className="inline-flex items-center gap-2 rounded-xl border border-white/[0.1] bg-white/[0.03] px-5 py-3 text-sm font-medium text-zinc-200 transition-colors hover:bg-white/[0.06]"
              >
                <IconChart className="h-4 w-4" />
                Robustness results
              </Link>
            </div>
          </div>
          <HeroDemo />
        </div>
      </section>

      {/* --------------------------------------------------------- stats */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Backbone"
          value="DINOv3 ViT-L/16"
          sub="full continuation fine-tuning"
          accent="cyan"
        />
        <StatCard
          label="Parameters"
          value="302M"
          sub="15% of the 2B budget"
          accent="sky"
        />
        <StatCard
          label="Input"
          value="512 × 512"
          sub="32 × 32 patch grid, small images upscaled"
          accent="emerald"
        />
        <StatCard
          label="Output"
          value="Verdict + heatmap"
          sub="global head + per-patch local head"
          accent="rose"
        />
      </section>

      {/* ----------------------------------------------------- pipeline */}
      <section>
        <SectionHeading
          eyebrow="Architecture"
          title="How a verdict is made"
          sub="One shared backbone, two heads — the global head answers “is this AI?”, the local head answers “where?”."
        />
        <FlowDiagram />
      </section>

      {/* --------------------------------------------------- evaluation */}
      <section>
        <SectionHeading
          eyebrow="Evaluation"
          title="Seer on held-out data"
          sub="runs/seer_vitl/last.pt (EMA), step 27,500 of 60,000 — clean protocol, threshold 0.5. 194,361 images across five held-out sets in 43 minutes on one RTX 4090; NTIRE val/test run inside the training loop."
        />
        <EvalResults />
      </section>

      {/* ------------------------------------------------ data mixture */}
      <section>
        <SectionHeading
          eyebrow="Training data"
          title="The mixture makes or breaks this task"
          sub="Ten public sources weighted by measured difficulty — 2.58M usable images, 1.70M fake and 875K real. Select a source for its contents and how to fetch it."
        />
        <MixtureTable />
      </section>

      {/* ------------------------------------------------------ footer */}
      <footer className="border-t border-white/[0.06] pt-8 text-xs text-zinc-600">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span>
            Seer · TikTok TechJam 2026 Track 5 · sub-2B AI-generated image
            detector
          </span>
          <span className="flex items-center gap-1.5">
            <IconZap className="h-3.5 w-3.5" />
            DINOv3 ViT-L · dual head · public data only
          </span>
        </div>
      </footer>
    </div>
  );
}

/* ---------------------------------------------------------------- pieces */

function SectionHeading({
  eyebrow,
  title,
  sub,
}: {
  eyebrow: string;
  title: string;
  sub?: string;
}) {
  return (
    <div className="max-w-2xl">
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-400/80">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
        {title}
      </h2>
      {sub && <p className="mt-3 text-sm leading-relaxed text-zinc-400">{sub}</p>}
    </div>
  );
}
