/** Display names for eval suite keys / filenames. */
const NAMES: Record<string, string> = {
  comfor_eval: "CommunityForensics-Eval",
  openfake_test: "OpenFake core/test",
  openfake_reddit: "OpenFake reddit/test",
  mirage: "MIRAGE",
  coco_val2017: "COCO val2017 (reals only)",
  ntire_test: "NTIRE 2026 public test",
  wildfake_dalle_advanced: "WildFake DALL·E Advanced",
};

const FAMILY: Record<string, string> = {
  LatDiff: "Latent diffusion",
  PixDiff: "Pixel diffusion",
  Commercial: "Commercial",
  GAN: "GAN",
  Other: "Other",
};

export const MIRAGE_META: Record<string, { name: string; desc: string }> = {
  RMG: {
    name: "Realistic model generation",
    desc: "Full-body e-commerce model shots — LoRA-tuned T2I renders the person, real garments composited in",
  },
  PCRMG: {
    name: "Pose-consistent model generation",
    desc: "RMG plus DWPose + ControlNet, keeping the original photo's pose",
  },
  T2I: {
    name: "Text-to-image",
    desc: "Vanilla outputs from generators unseen in the ID split — CogView4, Bagel, Wan2.1, HiDream, UniDiffuser",
  },
  IID: {
    name: "In-distribution, human-curated",
    desc: "Curated real + fake and ID-split T2I — the benchmark's ID test set",
  },
  "OOD-R": {
    name: "Out-of-distribution, human-curated",
    desc: "Expert-verified real + fake from a platform source the ID split never saw",
  },
  CB: {
    name: "Background replacement",
    desc: "Subject cut out; new background generated from the original caption",
  },
  TR: {
    name: "Virtual try-on",
    desc: "Clothing transferred between two model photos, then locally inpainted",
  },
  FS: {
    name: "Face swap",
    desc: "Faces exchanged between two real photos, then restored",
  },
  "IP/OP": {
    name: "Inpainting / outpainting",
    desc: "Masked regions regenerated, or the canvas extended and filled",
  },
  IE: {
    name: "Instruction-based editing",
    desc: "Natural-language edits from Flux-Kontext-class editors",
  },
};

export const HELDOUT_NOTES: Record<string, string> = {
  openfake_reddit:
    "In-the-wild test split only. Synthetic images are scraped from AI-generation subreddits, real images from photography subreddits — labels follow the subreddit, not the generator. Use this to evaluate how detectors trained on core transfer to naturally circulated content, with platform compression and unknown provenance.",
  coco_val2017:
    "Reals only — 5,000 COCO val2017 photographs. Macro accuracy and ranking metrics are not meaningful without fakes; the number that matters is FPR.",
  ntire_test:
    "NTIRE 2026 public test: 2,500 images, half tagged clean and half tagged distorted. On the open-test leaderboard in Gushchin et al. 2026, Table 3 (arXiv:2604.11487), Seer sits third on robust ROC AUC (0.9228), behind MICV and Ant International.",
  wildfake_dalle_advanced:
    "Fakes only — 8,843 DALL·E 3 Advanced images from WildFake. Macro accuracy is not meaningful without reals; read recall and FNR.",
};

export const HELDOUT_ORDER = [
  "comfor_eval",
  "openfake_test",
  "openfake_reddit",
  "mirage",
  "coco_val2017",
  "ntire_test",
  "wildfake_dalle_advanced",
];

export function evalKey(name?: string, file?: string): string {
  const fromFile = file?.split(/[\\/]/).pop()?.replace(/\.json$/i, "") ?? "";
  if (fromFile && fromFile in NAMES) return fromFile;
  if (name && name in NAMES) return name;
  return fromFile || name || "";
}

export function evalDisplayName(name?: string, file?: string): string {
  const key = evalKey(name, file);
  return NAMES[key] ?? name ?? key;
}

export function familyLabel(key: string): string {
  return FAMILY[key] ?? key;
}

export function finite(v: number | undefined | null): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}
