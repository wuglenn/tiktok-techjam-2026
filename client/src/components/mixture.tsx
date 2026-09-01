"use client";

import { useState } from "react";

import { Chip } from "@/components/essay";

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
    weight: 0.224,
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
    weight: 0.176,
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
    weight: 0.128,
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
    weight: 0.128,
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
    weight: 0.08,
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
    weight: 0.072,
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
    weight: 0.072,
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
    weight: 0.04,
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
    weight: 0.04,
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
    weight: 0.04,
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
    <div className="figure">
      {SOURCES.map((m) => {
        const isOpen = open === m.key;
        return (
          <div key={m.key} className="border-b border-dashed border-rule">
            <button
              type="button"
              onClick={() => setOpen(isOpen ? null : m.key)}
              aria-expanded={isOpen}
              className="grid w-full items-center gap-3 py-3 text-left sm:grid-cols-[1fr_140px_auto]"
            >
              <span className="flex min-w-0 flex-wrap items-center gap-2">
                <span className="text-[16px] font-medium text-ink-head">
                  {m.name}
                </span>
                <Chip>{m.cls}</Chip>
              </span>
              <span className="metric-bar">
                <span style={{ width: `${(m.weight / 0.224) * 100}%` }} />
              </span>
              <span className="tabular justify-self-end font-mono text-[14px] text-ink-mute">
                {m.weight.toFixed(3)}
              </span>
            </button>
            {isOpen && <SourceDetail m={m} />}
          </div>
        );
      })}
      <p className="caption py-3">
        usable by the mix: 2,576,437 images — 1,701,288 fake · 875,149 real.
      </p>
    </div>
  );
}

function SourceDetail({ m }: { m: MixtureSource }) {
  const fmt = (n: number) => n.toLocaleString("en-US");
  return (
    <div className="space-y-3 pb-4">
      <dl className="meta-grid">
        <div className="meta-pair">
          <dt>total</dt>
          <dd className="tabular">{fmt(m.total)}</dd>
        </div>
        {m.fake != null && (
          <div className="meta-pair">
            <dt>fake</dt>
            <dd className="tabular">{fmt(m.fake)}</dd>
          </div>
        )}
        {m.real != null && (
          <div className="meta-pair">
            <dt>real</dt>
            <dd className="tabular">{fmt(m.real)}</dd>
          </div>
        )}
        {m.generators && (
          <div className="meta-pair">
            <dt>generators</dt>
            <dd>{m.generators}</dd>
          </div>
        )}
      </dl>
      <p className="max-w-[600px] text-[16px] leading-[1.5] text-ink-body">{m.note}</p>
      <p className="font-mono text-[14px] leading-[1.4] text-ink-mute">{m.fetch}</p>
      {m.href && (
        <p>
          <a
            href={m.href}
            target="_blank"
            rel="noreferrer"
            className="ink-link underline decoration-1 underline-offset-[0.18em]"
          >
            dataset page
          </a>
        </p>
      )}
    </div>
  );
}
