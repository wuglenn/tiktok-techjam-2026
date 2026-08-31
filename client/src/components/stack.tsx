"use client";

import { useState } from "react";

import { IconExternal } from "@/components/icons";

/**
 * The submission's "built with" inventory — development tools, models/APIs,
 * libraries, and data — tabbed so the four deliverable answers live on one
 * page without burying the results above them.
 *
 * Kept in sync with project_description.md (sections 2-5) and pyproject.toml.
 */

interface StackItem {
  name: string;
  /** version pin or short tag, shown next to the name */
  meta?: string;
  role: string;
  href?: string;
}

interface StackGroup {
  label: string;
  items: StackItem[];
}

interface StackTab {
  key: string;
  label: string;
  hint: string;
  groups: StackGroup[];
  note?: string;
}

const TABS: StackTab[] = [
  {
    key: "tools",
    label: "Development tools",
    hint: "what the work was done in",
    groups: [
      {
        label: "Environment",
        items: [
          {
            name: "uv",
            meta: "Astral",
            role: "Python 3.10 environment and dependency resolution via pyproject.toml + uv.lock, with torch pinned to the cu124 index. Also the interpreter this dashboard spawns for live inference.",
            href: "https://docs.astral.sh/uv/",
          },
          {
            name: "Git + GitHub",
            role: "Version control and the public submission repository.",
          },
          {
            name: "PowerShell / bash",
            role: "Local Windows shell for development, remote Linux shell on the training pod.",
          },
        ],
      },
      {
        label: "Compute",
        items: [
          {
            name: "RunPod",
            role: "Training and full-scale evaluation — 40 hours of training on a H100 SXM with 251GB of RAM, with a /workspace network volume holding the ~2.5M-image mixture.",
            href: "https://runpod.io",
          },
          {
            name: "Hugging Face Hub CLI",
            role: "Gated-dataset and gated-weight authentication (hf auth login) plus shard fetching.",
          },
          {
            name: "HTTP range requests",
            meta: "eval_openfake/hfio.py",
            role: "Streams individual parquet row-groups straight out of the Hub, so a 67 GB evaluation split is scored image-by-image without ever landing on disk.",
          },
        ],
      },
      {
        label: "Verification and figures",
        items: [
          {
            name: "pytest",
            meta: "8 modules",
            role: "Offline test suite exercising the model, probe, optimizer, label mapping, and dataset adapters against a random tiny backbone — no network, no GPU, runs in seconds.",
          },
          {
            name: "matplotlib",
            role: "Heatmap overlay panels, error-analysis figures, and robustness charts.",
          },
          {
            name: "Next.js dev server",
            role: "This dashboard — the end-to-end demo surface, robustness summary, and error-analysis note.",
          },
        ],
      },
    ],
    note: "No Colab or Jupyter anywhere in the project. Every experiment is a CLI entry point (main.py train | eval | infer | info) driven by a YAML config, so any run is reproducible from one command and a config hash rather than from notebook cell order.",
  },
  {
    key: "models",
    label: "Models & APIs",
    hint: "backbone, heads, and external services",
    groups: [
      {
        label: "Backbone",
        items: [
          {
            name: "facebook/dinov3-vitl16-pretrain-lvd1689m",
            meta: "gated",
            role: "The backbone: DINOv3 ViT-L/16, 24 blocks, 1024-d, self-supervised on LVD-1689M. Fully fine-tuned for detection rather than used as a frozen feature extractor.",
            href: "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
          },
          {
            name: "camenduru/dinov3-vitl16-pretrain-lvd1689m",
            meta: "mirror",
            role: "Community mirror of the same weights, used on pods where the licence gate was not accepted.",
            href: "https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m",
          },
          {
            name: "facebook/dinov2-large · dinov2-small",
            meta: "fallback",
            role: "Ungated fallbacks in the same parameter class and a debug tier. The code path is backbone-agnostic and handles DINOv2 positional-encoding interpolation against DINOv3 RoPE.",
            href: "https://huggingface.co/facebook/dinov2-large",
          },
        ],
      },
      {
        label: "Heads — trained from scratch",
        items: [
          {
            name: "Global head",
            meta: "verdict",
            role: "LayerNorm(2048) → Linear(2048, 1024) → GELU → Dropout → Linear(1024, 1) over the concatenation of [CLS ; mean(patch tokens)], producing one image-level logit.",
          },
          {
            name: "Local head",
            meta: "heatmap",
            role: "Linear(1024, 1) applied per patch token → 1,024 logits on a 32×32 grid, sigmoid'd and bilinearly upsampled into the overlay.",
          },
          {
            name: "Probe heads",
            meta: "ablation",
            role: "LayerNorm + linear over a concatenation of four block taps (6/12/18/23), ~37k parameters — the frozen-feature baseline that measures whether full fine-tuning was necessary.",
          },
        ],
      },
      {
        label: "External APIs",
        items: [
          {
            name: "Hugging Face Hub HTTP API",
            role: "Dataset and weight resolution, repo_sha revision pinning, and byte-range parquet reads.",
          },
          {
            name: "CVDF S3",
            meta: "Open Images V7",
            role: "Real-image acquisition from the validation and test dumps.",
            href: "https://storage.googleapis.com/openimages/web/index.html",
          },
          {
            name: "No third-party inference APIs",
            meta: "by design",
            role: "Nothing is sent to a commercial detector or an LLM at train, eval, or inference time — the model runs entirely locally. Pangram Image's published numbers are quoted from their technical blog purely as a comparison target.",
          },
        ],
      },
    ],
    note: "Attention kernels are resolved at load time with graceful degradation: flash_attention_4 → flash_attention_3 → flash_attention_2 → sdpa, so the same checkpoint runs on a hackathon laptop and on an H100.",
  },
  {
    key: "libs",
    label: "Libraries & frameworks",
    hint: "Python and frontend dependencies",
    groups: [
      {
        label: "Python — pyproject.toml",
        items: [
          {
            name: "PyTorch",
            meta: "≥2.6 · cu124",
            role: "Training and inference — bf16 autocast, gradient checkpointing, binary_cross_entropy_with_logits, bilinear upsampling.",
          },
          {
            name: "Hugging Face Transformers",
            meta: "≥4.56",
            role: "AutoModel / AutoConfig loading of DINOv3 and DINOv2, plus attention-implementation selection.",
          },
          {
            name: "Hugging Face Datasets",
            meta: "≥3.2",
            role: "Streaming and local parquet reads for Community Forensics, FLUX-Reason-6M, SID_Set, and MIRAGE.",
          },
          {
            name: "huggingface-hub",
            meta: "≥0.30",
            role: "Authentication, repo revision pinning, and shard URL resolution.",
          },
          {
            name: "Pillow",
            meta: "≥10.4",
            role: "All image decode/encode and the entire augmentation stack — JPEG/WebP recompression, blur, resampling, filters.",
          },
          {
            name: "NumPy",
            meta: "≥1.26",
            role: "FFT-domain distortions (low-pass, phase noise), grain and speckle synthesis, heatmap arrays.",
          },
          {
            name: "scikit-learn",
            meta: "≥1.4",
            role: "roc_auc_score for AUROC and average_precision_score for mAP.",
          },
          {
            name: "matplotlib",
            meta: "≥3.8",
            role: "Heatmap overlays, error panels, robustness figures.",
          },
          {
            name: "PyYAML",
            meta: "≥6",
            role: "Config files with dotted --set overrides.",
          },
          {
            name: "tqdm",
            meta: "≥4.66",
            role: "Progress reporting on long evaluation sweeps.",
          },
          {
            name: "pyarrow",
            meta: "via datasets",
            role: "Parquet footer inspection and row-group-level reads for the streaming harness.",
          },
          {
            name: "diffusers · accelerate · sentencepiece · protobuf",
            meta: "optional",
            role: "The gen dependency group — only needed by scripts/generate_mirrors.py for synthetic mirroring.",
          },
        ],
      },
      {
        label: "Written here rather than imported",
        items: [
          {
            name: "Augmentation pipeline",
            meta: "~35 families",
            role: "Hand-written on Pillow and NumPy. The distortions that matter — 8×8 DCT grid shift, 4:2:0 chroma subsampling, resample-kernel mismatch, FFT phase noise — are in no standard transform library, and every one has to be reproducible from a seeded random.Random per sample.",
          },
          {
            name: "Muon optimizer",
            meta: "src/seer/optim.py",
            role: "Newton–Schulz-orthogonalized momentum on 2D weights with AdamW on everything else; used by the probe recipe.",
          },
          {
            name: "EMA, LLRD, schedule",
            role: "Exponential moving average, layer-wise learning-rate decay parameter groups, and a cosine-with-warmup schedule.",
          },
          {
            name: "Threaded prefetcher",
            meta: "BatchBuilder",
            role: "Decode pool profiled with scripts/bench_loader.py — eight decode threads roughly double collate throughput, which is what keeps an A100 fed.",
          },
        ],
      },
      {
        label: "Frontend",
        items: [
          {
            name: "Next.js",
            meta: "15 · App Router",
            role: "This dashboard, including the API routes that bridge into the Python model.",
          },
          { name: "React", meta: "19", role: "UI components." },
          {
            name: "TypeScript",
            meta: "5",
            role: "Strict types across components and API routes.",
          },
          {
            name: "Tailwind CSS",
            meta: "4",
            role: "Styling, via @tailwindcss/postcss.",
          },
          { name: "Geist", role: "Typeface." },
        ],
      },
    ],
    note: "Deliberately no torchvision, timm, or albumentations on the Python side, and no component or charting library on the frontend — the heatmap canvas, colormap, and charts are all local code.",
  },
  {
    key: "data",
    label: "Datasets & assets",
    hint: "training mixture, held-out sets, generated artifacts",
    groups: [
      {
        label: "Training mixture — 2,576,437 usable images, ~4,850 generators",
        items: [
          {
            name: "Ten weighted public sources",
            meta: "detailed above",
            role: "NTIRE 2026 train (0.28), CommunityForensics-Small (0.22), OpenFake core/train selected by measured difficulty (0.16), LAION-400M (0.16), GAS-Station v4 (0.10) and v3 (0.09), Open Images V7 (0.09), FLUX-Reason-6M (0.05), SID_Set (0.05), frontier fakes (0.05). Expand any row in the mixture table above for counts, generators, and its fetch command.",
          },
          {
            name: "Real-image breadth",
            meta: "5 pipelines",
            role: "Reals come from NTIRE 31%, Community Forensics 24%, LAION 18%, OpenFake (Pexels + ReLAION) 18%, Open Images 10% — five different capture and curation pipelines, which is what keeps the false-positive rate flat when the real distribution shifts.",
          },
        ],
      },
      {
        label: "Held-out evaluation — unreachable by the training loader",
        items: [
          {
            name: "CommunityForensics-Eval",
            meta: "51,836",
            role: "21 generators, balanced 25,918/25,918 — the Pangram evaluation protocol with a per-architecture breakdown.",
            href: "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small",
          },
          {
            name: "OpenFake core/test",
            meta: "89,225",
            role: "20 generators absent from the mixture entirely, scored against reals it has also never seen (DOCCI + ImageNet) — unseen generators and unseen reals simultaneously.",
            href: "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
          },
          {
            name: "OpenFake reddit/test",
            meta: "36,227",
            role: "In the wild — AI subreddits against photography subreddits, provenance unknown.",
          },
          {
            name: "MIRAGE",
            meta: "12,073",
            role: "Human-verified in-the-wild set including inpainting, face-swap, and image-edit slices.",
          },
          {
            name: "NTIRE 2026 val · val-hard · public test",
            meta: "10,000 / 2,500 / 2,500",
            role: "The clean-versus-distorted and per-distortion robustness protocol.",
          },
          {
            name: "COCO val2017 · WikiArt",
            meta: "5,000 reals",
            role: "False-positive-only harnesses — the organisers' reference real half, plus digital art as the hardest real class.",
          },
        ],
      },
      {
        label: "Generated assets in this repo",
        items: [
          {
            name: "best.pt",
            meta: "~4.9 GB",
            role: "Trained checkpoint — model weights, EMA shadow, and optimizer state.",
          },
          {
            name: "eval_openfake/out/full_core_test/",
            meta: "91,398 rows",
            role: "The full OpenFake core/test sweep: rows.jsonl with a per-image score, aggregate.json with metrics plus per-generator and per-real-source breakdowns, and 48 error-panel PNGs.",
          },
          {
            name: "eval_openfake/out/panels/",
            meta: "10 panels",
            role: "Curated four-row comparison panels used in the write-up and demo.",
          },
          {
            name: "docs/deliverables/heldout-eval-step27500.md",
            role: "The full held-out suite report for the strongest checkpoint.",
          },
          {
            name: "Heatmap PNGs",
            meta: "turbo · 0.55α",
            role: "Rendered by src/seer/heatmap.py from the local head's patch logits.",
          },
        ],
      },
    ],
    note: "Everything is public. The loader hard-refuses any path under openfake/holdout_*, core/test, reddit/test, comfor-eval, or COCO val2017, so a held-out image cannot leak into training through a config mistake. Community Forensics is CC BY-NC-SA 4.0 and OpenFake is CC-BY-SA-4.0 with non-commercial restrictions on the proprietary-generator subsets — this project is a non-commercial research artifact accordingly, and redistributes no third-party media.",
  },
];

export function StackTabs() {
  const [tab, setTab] = useState(0);
  const active = TABS[tab];

  return (
    <div className="mt-8">
      <div className="flex flex-wrap gap-2">
        {TABS.map((t, i) => (
          <button
            key={t.key}
            onClick={() => setTab(i)}
            title={t.hint}
            aria-pressed={i === tab}
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

      <div key={active.key} className="animate-rise mt-5 space-y-6">
        {active.groups.map((g) => (
          <div key={g.label}>
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              {g.label}
            </p>
            <div className="panel mt-2.5 divide-y divide-white/[0.05]">
              {g.items.map((it) => (
                <Row key={it.name} it={it} />
              ))}
            </div>
          </div>
        ))}

        {active.note && (
          <p className="max-w-3xl text-xs leading-relaxed text-zinc-500">
            {active.note}
          </p>
        )}
      </div>
    </div>
  );
}

function Row({ it }: { it: StackItem }) {
  return (
    <div className="grid gap-1.5 px-5 py-3.5 sm:grid-cols-[minmax(0,15rem)_1fr] sm:gap-6">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {it.href ? (
          <a
            href={it.href}
            target="_blank"
            rel="noreferrer"
            className="group inline-flex items-center gap-1.5 text-sm font-medium text-zinc-100 transition-colors hover:text-cyan-300"
          >
            <span className="break-words">{it.name}</span>
            <IconExternal className="h-3 w-3 shrink-0 text-zinc-600 transition-colors group-hover:text-cyan-300" />
          </a>
        ) : (
          <span className="break-words text-sm font-medium text-zinc-100">
            {it.name}
          </span>
        )}
        {it.meta && (
          <span className="tabular shrink-0 rounded-full border border-white/[0.08] bg-white/[0.03] px-2 py-0.5 text-[10px] font-medium text-zinc-400">
            {it.meta}
          </span>
        )}
      </div>
      <p className="text-xs leading-relaxed text-zinc-400">{it.role}</p>
    </div>
  );
}
