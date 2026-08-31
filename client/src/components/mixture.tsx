"use client";

import { useState } from "react";

import { IconChevron, IconExternal } from "@/components/icons";

/**
 * The hero training mixture. Counts are the on-disk inventory from
 * docs/DATA_MIXTURE.md (weights decide draw probability, not disk usage);
 * links go to each dataset's public page.
 */
interface MixtureSource {
  key: string;
  name: string;
  cls: "real" | "fake" | "mixed";
  weight: number;
  fake?: number;
  real?: number;
  total: number;
  generators?: string;
  note: string;
  fetch: string;
  href?: string;
}

const SOURCES: MixtureSource[] = [
  {
    key: "ntire",
    name: "NTIRE 2026 train",
    cls: "mixed",
    weight: 0.28,
    fake: 177643,
    real: 100000,
    total: 277643,
    generators: "42 (2022–2026, not tagged per image)",
    note: "All six shards of the NTIRE 2026 challenge training set, real/fake matched per shard — the mixture's recency anchor.",
    fetch: "python get_datasets.py --only ntire-train",
  },
  {
    key: "comfor",
    name: "CommunityForensics-Small",
    cls: "mixed",
    weight: 0.22,
    fake: 278445,
    real: 278096,
    total: 556541,
    generators: "4,782 — 19 named + 4,763 HF community",
    note: "Paired real/fake images from open generators; the strongest driver of unseen-architecture transfer. The CommunityForensics-Eval split is held out for evaluation.",
    fetch: "scripts/fetch_data.py comfor-small",
    href: "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small",
  },
  {
    key: "openfake",
    name: "OpenFake (selected)",
    cls: "mixed",
    weight: 0.16,
    fake: 309523,
    real: 130000,
    total: 439523,
    generators: "30 ranked by measured recall",
    note: "Frontier/community generators the detector measurably misses — fetched inversely to recall (worst generators get the most images) — plus LAION and Pexels reals.",
    fetch: "scripts/openfake.py fetch --from-rank …",
    href: "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
  },
  {
    key: "laion400m-1",
    name: "laion400m-1",
    cls: "real",
    weight: 0.16,
    real: 199998,
    total: 199998,
    note: "Hosted LAION-400M images in parquet (not a URL scrape), size-filtered to >512px per side; growing toward 400k.",
    fetch: "scripts/download_laion400m.py",
    href: "https://huggingface.co/datasets/jp1924/Laion400m-1",
  },
  {
    key: "gs-images-v4",
    name: "GAS-Station v4",
    cls: "fake",
    weight: 0.1,
    fake: 113793,
    total: 113793,
    generators: "15 model folders (58,733 unlabeled)",
    note: "Later weekly open-model miner dumps with model_name tags (FLUX, SDXL, Janus, CogView, …).",
    fetch: "scripts/wire_gasstation.py --versions v4",
    href: "https://huggingface.co/datasets/gasstation/gs-images-v4",
  },
  {
    key: "gs-images-v3",
    name: "GAS-Station v3",
    cls: "fake",
    weight: 0.09,
    fake: 426689,
    total: 426689,
    generators: "19 model folders (186,579 unlabeled)",
    note: "Weekly open-model miner dumps with model_name tags.",
    fetch: "scripts/wire_gasstation.py --versions v3",
    href: "https://huggingface.co/datasets/gasstation/gs-images-v3",
  },
  {
    key: "open-images-v7",
    name: "Open Images V7",
    cls: "real",
    weight: 0.09,
    real: 167055,
    total: 167055,
    note: "Web photographs from the CVDF S3 dump (validation + test splits).",
    fetch: "scripts/download_open_images.py",
    href: "https://storage.googleapis.com/openimages/web/index.html",
  },
  {
    key: "flux-reason",
    name: "FLUX-Reason-6M",
    cls: "fake",
    weight: 0.05,
    fake: 320000,
    total: 320000,
    generators: "FLUX.1-dev only",
    note: "320k local slice of the ~5.9M-image dump, read in streaming mode rather than snapshotted (~882 GB full).",
    fetch: "scripts/fetch_data.py flux-reason-6m --max-shards 8",
    href: "https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M",
  },
  {
    key: "frontier-fakes",
    name: "Frontier fakes",
    cls: "fake",
    weight: 0.05,
    fake: 5195,
    total: 10695,
    generators: "untagged Midjourney / DALL-E / SD / Nano Banana Pro",
    note: "Labelled frontier-generator set; 5,195 fakes used of the 10,695 in the set.",
    fetch: "scripts/fetch_data.py frontier-fakes",
    href: "https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
  },
  {
    key: "sid-set",
    name: "SID_Set",
    cls: "fake",
    weight: 0.05,
    fake: 70000,
    total: 210000,
    generators: "untagged (full-synthetic only)",
    note: "Full-synthetic social-media images — 70,000 class-1 of 210,000; the real and tampered splits are dropped.",
    fetch: "scripts/fetch_data.py sid-set --max-shards 16",
    href: "https://huggingface.co/datasets/saberzl/SID_Set",
  },
];

export function MixtureTable() {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="panel mt-8 divide-y divide-white/[0.05]">
      {SOURCES.map((m) => {
        const isOpen = open === m.key;
        return (
          <div key={m.key}>
            <button
              onClick={() => setOpen(isOpen ? null : m.key)}
              aria-expanded={isOpen}
              className="grid w-full items-center gap-3 px-5 py-3 text-left transition-colors hover:bg-white/[0.02] sm:grid-cols-[1fr_160px_auto]"
            >
              <span className="flex min-w-0 items-center gap-3">
                <IconChevron
                  className={`h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform duration-200 ${
                    isOpen ? "rotate-180" : ""
                  }`}
                />
                <span className="truncate text-sm font-medium text-zinc-100">
                  {m.name}
                </span>
                <span
                  className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                    m.cls === "real"
                      ? "bg-emerald-500/10 text-emerald-300"
                      : m.cls === "fake"
                        ? "bg-rose-500/10 text-rose-300"
                        : "bg-cyan-500/10 text-cyan-300"
                  }`}
                >
                  {m.cls}
                </span>
              </span>
              <span className="h-1.5 overflow-hidden rounded-full bg-white/[0.05]">
                <span
                  className="block h-full rounded-full bg-cyan-400/50"
                  style={{ width: `${(m.weight / 0.28) * 100}%` }}
                />
              </span>
              <span className="tabular justify-self-end text-xs font-medium text-zinc-300">
                {m.weight.toFixed(2)}
              </span>
            </button>
            {isOpen && <SourceDetail m={m} />}
          </div>
        );
      })}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-white/[0.02] px-5 py-3 text-xs text-zinc-500">
        <span>
          usable by the mix:{" "}
          <span className="tabular font-semibold text-zinc-200">2,576,437</span>{" "}
          images — 1,701,288 fake · 875,149 real
        </span>
        <span>weights sum to 1.25</span>
      </div>
    </div>
  );
}

function SourceDetail({ m }: { m: MixtureSource }) {
  const fmt = (n: number) => n.toLocaleString("en-US");
  return (
    <div className="animate-rise space-y-4 px-5 pb-5 pt-1 sm:pl-10">
      <div className="flex flex-wrap gap-x-10 gap-y-3">
        <Stat label="total" value={fmt(m.total)} />
        {m.fake != null && <Stat label="fake" value={fmt(m.fake)} tone="rose" />}
        {m.real != null && <Stat label="real" value={fmt(m.real)} tone="emerald" />}
        {m.generators && <Stat label="generators" value={m.generators} />}
      </div>
      <p className="max-w-2xl text-xs leading-relaxed text-zinc-400">{m.note}</p>
      <div className="flex flex-wrap items-center gap-3">
        <code className="rounded-lg border border-white/[0.06] bg-black/40 px-2.5 py-1 text-[11px] text-zinc-300">
          {m.fetch}
        </code>
        {m.href && (
          <a
            href={m.href}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-[11px] font-medium text-cyan-300 transition-colors hover:text-cyan-200"
          >
            <IconExternal className="h-3.5 w-3.5" />
            dataset page
          </a>
        )}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "rose" | "emerald";
}) {
  return (
    <div>
      <p className="text-[10px] font-medium uppercase tracking-[0.16em] text-zinc-500">
        {label}
      </p>
      <p
        className={`tabular mt-0.5 text-sm font-semibold ${
          tone === "rose"
            ? "text-rose-300"
            : tone === "emerald"
              ? "text-emerald-300"
              : "text-white"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
