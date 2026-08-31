"use client";

import { useState } from "react";

import { HeatCanvas } from "@/components/heat-canvas";
import { IconEye } from "@/components/icons";
import { ProbGauge, VerdictPill } from "@/components/verdict";
import { syntheticGrid } from "@/lib/heat";

const GRID = syntheticGrid(1337, 0.987);
const PROB = 0.987;

/** Decorative but real-rendered verdict card for the hero (demo sample). */
export function HeroDemo() {
  const [showHeat, setShowHeat] = useState(true);
  return (
    <div className="relative">
      <div className="absolute -inset-8 -z-10 rounded-[32px] bg-[radial-gradient(closest-side,rgba(34,211,238,0.1),transparent)] blur-2xl" />
      <div className="panel animate-rise overflow-hidden p-4 sm:p-5">
        <div className="flex items-center justify-between gap-3">
          <VerdictPill label="AI" />
          <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-[10px] font-medium tracking-wide text-zinc-400">
            sample verdict
          </span>
        </div>

        <div className="relative mt-4 aspect-[4/3] overflow-hidden rounded-xl ring-1 ring-white/10">
          <HeatCanvas grid={GRID} showHeat={showHeat} opacity={0.55} />
          <button
            onClick={() => setShowHeat((v) => !v)}
            title="toggle heatmap"
            className={`absolute bottom-2.5 right-2.5 flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-[10px] font-medium backdrop-blur-md transition-colors ${
              showHeat
                ? "bg-cyan-500/85 text-zinc-950"
                : "bg-black/60 text-zinc-300 hover:text-white"
            }`}
          >
            <IconEye className="h-3.5 w-3.5" />
            heatmap
          </button>
        </div>

        <div className="mt-4 flex items-center justify-between gap-4">
          <ProbGauge p={PROB} size={118} />
          <div className="space-y-1.5 text-right">
            <p className="text-xs font-medium text-zinc-300">flux.1-dev</p>
            <p className="text-[10px] text-zinc-500">
              32×32 patch grid · threshold 0.5
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
