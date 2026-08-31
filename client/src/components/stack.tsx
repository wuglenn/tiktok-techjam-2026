"use client";

import { useState } from "react";

import { Chip, Measure, Tabs } from "@/components/essay";

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
            name: "RunPod",
            meta: "1× H100 SXM",
            role: "40 hours of training on 16 vCPU (Xeon Platinum 8462Y+), 251 GB RAM, 30 GB container disk, and a 2 TB network volume at /workspace holding the 2.5M-image mixture.",
            href: "https://runpod.io",
          },
        ],
      },
    ],
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
            role: "DINOv3 ViT-L/16, 24 blocks, 1024-d, self-supervised on LVD-1689M. Fully fine-tuned for detection rather than used as a frozen feature extractor.",
            href: "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
          },
        ],
      },
      {
        label: "Heads, trained from scratch",
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
        ],
      }
    ],
    note: "Attention kernels resolve at load time with graceful degradation (flash_attention_4 → 3 → 2 → sdpa), so the same checkpoint runs on a laptop and on an H100.",
  },
  {
    key: "data",
    label: "Datasets & assets",
    hint: "training mixture, held-out sets, generated artifacts",
    groups: [
      {
        label: "Training mixture: 2,576,437 usable images, ~4,850 generators",
        items: [
          {
            name: "Ten weighted public sources",
            meta: "detailed above",
            role: "NTIRE 2026 train (0.28), CommunityForensics-Small (0.22), OpenFake core/train selected by measured difficulty (0.16), LAION-400M (0.16), GAS-Station v4 (0.10) and v3 (0.09), Open Images V7 (0.09), FLUX-Reason-6M (0.05), SID_Set (0.05), frontier fakes (0.05).",
          },
          {
            name: "Real-image breadth",
            meta: "5 pipelines",
            role: "Reals come from NTIRE 31%, Community Forensics 24%, LAION 18%, OpenFake 18%, Open Images 10%. Five capture and curation pipelines, which is what keeps the false-positive rate flat when the real distribution shifts.",
          },
        ],
      },
      {
        label: "Held-out evaluation, unreachable by the training loader",
        items: [
          {
            name: "CommunityForensics-Eval",
            meta: "51,836",
            role: "21 generators, balanced 25,918/25,918. The CompEval protocol, with a per-architecture breakdown.",
            href: "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small",
          },
          {
            name: "OpenFake core/test",
            meta: "89,225",
            role: "20 generators absent from the mixture entirely, scored against reals it has also never seen (DOCCI + ImageNet).",
            href: "https://huggingface.co/datasets/ComplexDataLab/OpenFake",
          },
          {
            name: "OpenFake reddit/test",
            meta: "36,227",
            role: "In the wild: AI subreddits against photography subreddits, provenance unknown.",
          },
          {
            name: "MIRAGE",
            meta: "12,073",
            role: "Human-verified in-the-wild set, including inpainting, face-swap, and image-edit slices.",
          },
          {
            name: "NTIRE 2026 public test",
            meta: "2,500",
            role: "The clean-versus-distorted protocol, and the published open-test leaderboard where Seer sits third on robust ROC AUC.",
          },
          {
            name: "COCO val2017",
            meta: "5,000 reals",
            role: "False-positive-only harness: the organisers' reference real photographs.",
          },
          {
            name: "WildFake DALL·E Advanced",
            meta: "8,843 fakes",
            role: "DALL·E 3 Advanced stills only — recall, and the first false-negative panel on the Errors page.",
          },
        ],
      },
      {
        label: "Generated assets in this repo",
        items: [
          {
            name: "best.pt",
            meta: "~4.9 GB",
            role: "Trained checkpoint: model weights, EMA shadow, and optimizer state.",
          }
        ],
      },
    ],
    note: "Everything is public. The loader hard-refuses any path under openfake/holdout_*, core/test, reddit/test, comfor-eval, or COCO val2017, so a held-out image cannot leak into training through a config mistake. Community Forensics is CC BY-NC-SA 4.0 and OpenFake CC-BY-SA-4.0 with non-commercial restrictions, so this is a non-commercial research artifact; no third-party media is redistributed.",
  },
];
export function StackTabs() {
  const [tab, setTab] = useState(0);
  const active = TABS[tab];

  return (
    <div className="figure">
      <Tabs
        items={TABS.map((t) => ({ label: t.label, hint: t.hint }))}
        active={tab}
        onChange={setTab}
      />

      <div className="mt-6 space-y-8">
        {active.groups.map((g) => (
          <div key={g.label}>
            <p className="small-head">{g.label}</p>
            <div className="mt-2">
              {g.items.map((it) => (
                <Row key={it.name} it={it} />
              ))}
            </div>
          </div>
        ))}

        {active.note && (
          <Measure className="!px-0">
            <p className="text-[16px] leading-[1.5] text-ink-body">{active.note}</p>
          </Measure>
        )}
      </div>
    </div>
  );
}

function Row({ it }: { it: StackItem }) {
  return (
    <div className="grid gap-1.5 border-b border-dashed border-rule py-3 sm:grid-cols-[minmax(0,15rem)_1fr] sm:gap-6">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {it.href ? (
          <a
            href={it.href}
            target="_blank"
            rel="noreferrer"
            className="ink-link break-words text-[16px] font-medium underline decoration-1 underline-offset-[0.18em]"
          >
            {it.name}
          </a>
        ) : (
          <span className="break-words text-[16px] font-medium text-ink-head">
            {it.name}
          </span>
        )}
        {it.meta && <Chip>{it.meta}</Chip>}
      </div>
      <p className="text-[16px] leading-[1.5] text-ink-body">{it.role}</p>
    </div>
  );
}
