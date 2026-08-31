"use client";

import { useState } from "react";

import { MetricBar } from "@/components/charts";
import { pct } from "@/lib/format";

/**
 * Per-set detail from docs/deliverables/heldout-eval-step27500.md, tabbed so
 * the overview stays scannable: CommunityForensics generator families,
 * OpenFake core/test per-generator, and MIRAGE per-source.
 */

const TABS = [
  { label: "Community Forensics", hint: "CommunityForensics-Eval by architecture family" },
  { label: "OpenFake", hint: "core/test, fake-only buckets" },
  { label: "MIRAGE", hint: "human-verified in-the-wild" },
] as const;

const COMPFOR_FAMILIES = [
  { family: "Other", n: "2,000", acc: 0.999, recall: 0.998, map: 1.0, fpr: 0.0 },
  { family: "Latent diffusion", n: "12,000", acc: 0.9968, recall: 0.995, map: 1.0, fpr: 0.0013 },
  { family: "GAN", n: "4,000", acc: 0.994, recall: 0.989, map: 1.0, fpr: 0.001 },
  { family: "Commercial", n: "29,836", acc: 0.9444, recall: 0.8912, map: 0.9955, fpr: 0.0024 },
  { family: "Pixel diffusion", n: "4,000", acc: 0.867, recall: 0.7345, map: 0.9884, fpr: 0.0005 },
];

const OPENFAKE_GENERATORS = [
  { gen: "illustrious", n: "6,694", recall: 0.9988 },
  { gen: "seedream-v5.0", n: "372", recall: 0.9973 },
  { gen: "aurora-20-1-25", n: "282", recall: 0.9965 },
  { gen: "lumina-17-2-25", n: "543", recall: 0.9963 },
  { gen: "ernie-image-turbo", n: "687", recall: 0.9927 },
  { gen: "gpt-image-1.5", n: "5,573", recall: 0.9837 },
  { gen: "flux.2-klein-9b", n: "8,249", recall: 0.9736 },
  { gen: "wan-video-2.5", n: "1,174", recall: 0.9719 },
  { gen: "ernie-image", n: "315", recall: 0.9651 },
  { gen: "z-image-turbo", n: "12,634", recall: 0.9529 },
  { gen: "recraft-v2", n: "282", recall: 0.9362 },
  { gen: "gpt-image-2", n: "474", recall: 0.9304 },
  { gen: "veo-3", n: "2,167", recall: 0.9262 },
  { gen: "nano-banana-pro", n: "386", recall: 0.9171 },
  { gen: "sora-2", n: "557", recall: 0.8977 },
  { gen: "midjourney-7", n: "3,586", recall: 0.8332 },
  { gen: "ideogram-2.0", n: "282", recall: 0.7589 },
  { gen: "frames-23-1-25", n: "250", recall: 0.756 },
  { gen: "halfmoon-4-4-25", n: "190", recall: 0.7211 },
  { gen: "recraft-v3", n: "1,000", recall: 0.569 },
];

const MIRAGE_SOURCES = [
  {
    source: "RMG",
    name: "Realistic model generation",
    desc: "Full-body e-commerce model shots — LoRA-tuned T2I renders the person, real garments composited in",
    n: "2,499 / 0",
    acc: 0.9888,
    recall: 0.9888,
  },
  {
    source: "PCRMG",
    name: "Pose-consistent model generation",
    desc: "RMG plus DWPose + ControlNet, keeping the original photo's pose",
    n: "565 / 0",
    acc: 0.9805,
    recall: 0.9805,
  },
  {
    source: "T2I",
    name: "Text-to-image",
    desc: "Vanilla outputs from generators unseen in the ID split — CogView4, Bagel, Wan2.1, HiDream, UniDiffuser",
    n: "3,391 / 0",
    acc: 0.9106,
    recall: 0.9106,
  },
  {
    source: "IID",
    name: "In-distribution, human-curated",
    desc: "Curated real + fake and ID-split T2I — the benchmark's ID test set",
    n: "883 / 798",
    acc: 0.837,
    recall: 0.7293,
    fpr: 0.0439,
  },
  {
    source: "OOD-R",
    name: "Out-of-distribution, human-curated",
    desc: "Expert-verified real + fake from a platform source the ID split never saw",
    n: "609 / 593",
    acc: 0.782,
    recall: 0.6388,
    fpr: 0.0708,
  },
  {
    source: "CB",
    name: "Background replacement",
    desc: "Subject cut out; new background generated from the original caption",
    n: "286 / 0",
    acc: 0.535,
    recall: 0.535,
  },
  {
    source: "TR",
    name: "Virtual try-on",
    desc: "Clothing transferred between two model photos, then locally inpainted",
    n: "427 / 0",
    acc: 0.4707,
    recall: 0.4707,
  },
  {
    source: "FS",
    name: "Face swap",
    desc: "Faces exchanged between two real photos, then restored",
    n: "218 / 0",
    acc: 0.445,
    recall: 0.445,
  },
  {
    source: "IP/OP",
    name: "Inpainting / outpainting",
    desc: "Masked regions regenerated, or the canvas extended and filled",
    n: "990 / 0",
    acc: 0.4283,
    recall: 0.4283,
  },
  {
    source: "IE",
    name: "Instruction-based editing",
    desc: "Natural-language edits from Flux-Kontext-class editors",
    n: "814 / 0",
    acc: 0.3894,
    recall: 0.3894,
  },
];

export function EvalBreakdowns() {
  const [tab, setTab] = useState(0);

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {TABS.map((t, i) => (
          <button
            key={t.label}
            onClick={() => setTab(i)}
            title={t.hint}
            className={`rounded-full px-4 py-1.5 text-xs font-medium transition-colors ${
              i === tab
                ? "bg-cyan-400 text-zinc-950"
                : "border border-white/[0.08] bg-white/[0.03] text-zinc-400 hover:text-zinc-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mt-4">
        {tab === 0 && <CompforFamilies />}
        {tab === 1 && <OpenfakeGenerators />}
        {tab === 2 && <MirageSources />}
      </div>
    </div>
  );
}

function CompforFamilies() {
  return (
    <div>
      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                <th className="px-5 py-3 font-medium">Family</th>
                <th className="px-4 py-3 font-medium">n</th>
                <th className="px-4 py-3 font-medium">Acc</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-4 py-3 font-medium">mAP</th>
                <th className="px-5 py-3 font-medium">FPR</th>
              </tr>
            </thead>
            <tbody>
              {COMPFOR_FAMILIES.map((f, i) => (
                <tr
                  key={f.family}
                  className={`transition-colors hover:bg-white/[0.03] ${
                    i < COMPFOR_FAMILIES.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <td className="px-5 py-3 font-medium text-zinc-100">{f.family}</td>
                  <td className="tabular px-4 py-3 text-zinc-400">{f.n}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2.5">
                      <span className="tabular w-12 font-medium text-white">{pct(f.acc)}</span>
                      <div className="w-20">
                        <MetricBar value={f.acc} />
                      </div>
                    </div>
                  </td>
                  <td className="tabular px-4 py-3 text-zinc-300">{pct(f.recall)}</td>
                  <td className="tabular px-4 py-3 text-zinc-300">{pct(f.map)}</td>
                  <td className="tabular px-5 py-3 text-emerald-300/90">{pct(f.fpr)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function OpenfakeGenerators() {
  return (
    <div>
      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                <th className="px-5 py-3 font-medium">Generator</th>
                <th className="px-4 py-3 font-medium">n</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-5 py-3 font-medium">FNR</th>
              </tr>
            </thead>
            <tbody>
              {OPENFAKE_GENERATORS.map((g, i) => (
                <tr
                  key={g.gen}
                  className={`transition-colors hover:bg-white/[0.03] ${
                    i < OPENFAKE_GENERATORS.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <td className="px-5 py-2.5 font-medium text-zinc-100">{g.gen}</td>
                  <td className="tabular px-4 py-2.5 text-zinc-400">{g.n}</td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2.5">
                      <span className="tabular w-12 font-medium text-white">{pct(g.recall)}</span>
                      <div className="w-24">
                        <MetricBar value={g.recall} />
                      </div>
                    </div>
                  </td>
                  <td className="tabular px-5 py-2.5 text-amber-300/90">
                    {pct(1 - g.recall)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="mt-3 max-w-3xl text-xs leading-relaxed text-zinc-500">
        Fake-only buckets — no reals per bucket, so precision is 1.0 and
        accuracy equals recall. Held-out generators <em className="not-italic text-zinc-300">and</em>{" "}
        held-out reals: DOCCI 0.36% FPR (14,847 images) and ImageNet 0.13% FPR
        (28,681). The hole is a small set of stylized generators — recraft-v3,
        halfmoon, frames, ideogram-2, midjourney-7 — not the GPT Image / FLUX.2 /
        Seedream mass.
      </p>
    </div>
  );
}

function MirageSources() {
  return (
    <div>
      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-[11px] uppercase tracking-[0.12em] text-zinc-500">
                <th className="px-5 py-3 font-medium">Source</th>
                <th className="px-4 py-3 font-medium">n (fake / real)</th>
                <th className="px-4 py-3 font-medium">Acc</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-5 py-3 font-medium">FPR</th>
              </tr>
            </thead>
            <tbody>
              {MIRAGE_SOURCES.map((s, i) => (
                <tr
                  key={s.source}
                  className={`transition-colors hover:bg-white/[0.03] ${
                    i < MIRAGE_SOURCES.length - 1 ? "border-b border-white/[0.04]" : ""
                  }`}
                >
                  <td className="px-5 py-2.5">
                    <div className="font-medium text-zinc-100">
                      {s.name}{" "}
                      <span className="text-[11px] font-normal text-zinc-500">{s.source}</span>
                    </div>
                    <div className="mt-0.5 max-w-[260px] text-[11px] leading-snug text-zinc-500">
                      {s.desc}
                    </div>
                  </td>
                  <td className="tabular px-4 py-2.5 text-zinc-400">{s.n}</td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2.5">
                      <span className="tabular w-12 font-medium text-white">{pct(s.acc)}</span>
                      <div className="w-20">
                        <MetricBar value={s.acc} />
                      </div>
                    </div>
                  </td>
                  <td className="tabular px-4 py-2.5 text-zinc-300">{pct(s.recall)}</td>
                  <td className="tabular px-5 py-2.5 text-emerald-300/90">
                    {s.fpr != null ? pct(s.fpr) : <span className="text-zinc-600">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="mt-3 max-w-3xl text-xs leading-relaxed text-zinc-500">
        MIRAGE tags each image by how it was built, not by which generator
        built it. Full-image synthesis is largely solved — text-to-image and
        the two composite model-shot pipelines recall at 91–99%. The hole is
        local edits: instruction editing, inpainting/outpainting, face swap,
        try-on and background swap sit at 39–54% — exactly the composite-like
        edits the patch head is meant to catch, but this pass is page-level
        only. The two mixed real/fake slices are the human-curated ones, and
        that is where the 4–7% FPR lives.
      </p>
    </div>
  );
}
