"""Single registry of every dataset in the plan.

Both ``get_datasets.py`` (acquisition) and ``dataset_stats.py`` (remote
inspection) read from here, so sizes, licences and caveats cannot drift apart
between the two.

Tiers reflect value-per-gigabyte for this task rather than dataset quality:

tier 1  everything needed to build, train and evaluate end to end
tier 2  generator breadth and extra held-out benchmarks
tier 3  large or awkward, only worth it with spare bandwidth
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Source = Literal["hf", "hf_files", "url", "kaggle", "modelscope", "generate", "manual"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    source: Source
    tier: int
    role: str                       # what we use it for
    repo_id: str = ""               # HF repo / kaggle slug / modelscope id
    repo_type: str = "dataset"
    files: tuple[str, ...] = ()     # specific files, empty means whole repo
    url: str = ""
    approx_gb: float = 0.0
    licence: str = "unknown"
    homepage: str = ""
    notes: tuple[str, ...] = ()
    # Populated for HF datasets whose metadata columns let us audit
    # real/fake confounds remotely (format, resolution, label).
    stat_columns: tuple[str, ...] = ()
    config: str = "default"
    split: str = "train"


NTIRE_TRAIN = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
NTIRE_VAL = "deepfakesMSU/NTIRE-RobustAIGenDetection-val"
NTIRE_TEST = "deepfakesMSU/NTIRE-RobustAIGenDetection-test-public"

REGISTRY: tuple[DatasetSpec, ...] = (
    # ---------------- tier 1 : core ----------------
    DatasetSpec(
        key="ntire-val",
        name="NTIRE 2026 validation (labelled)",
        source="hf_files",
        tier=1,
        role="Primary held-out evaluation. Labels include per-image distortion type and severity.",
        repo_id=NTIRE_VAL,
        files=("val_labels.csv", "val_hard_labels.csv", "val_images.zip", "val_images_hard.zip"),
        approx_gb=4.0,
        licence="untagged",
        homepage=f"https://huggingface.co/datasets/{NTIRE_VAL}",
        notes=(
            "README claims no labels; the label CSVs are present and valid.",
            "Half of each class is degraded; the other half is clean.",
        ),
    ),
    DatasetSpec(
        key="ntire-test",
        name="NTIRE 2026 public test (labelled)",
        source="hf_files",
        tier=1,
        role="Second held-out set, unseen proprietary generators.",
        repo_id=NTIRE_TEST,
        files=("test_labels.csv", "test_images.zip"),
        approx_gb=0.85,
        licence="untagged",
        homepage=f"https://huggingface.co/datasets/{NTIRE_TEST}",
    ),
    DatasetSpec(
        key="ntire-train",
        name="NTIRE 2026 train shard 0",
        source="hf_files",
        tier=1,
        role="Primary training source: 42 generators (2022-2026), real/fake matched on resolution, aspect ratio and JPEG quality.",
        repo_id=NTIRE_TRAIN,
        files=("shard_0.zip",),
        approx_gb=19.0,
        licence="untagged",
        homepage=f"https://huggingface.co/datasets/{NTIRE_TRAIN}",
        notes=(
            "6 shards of ~50k, all distribution-matched, so one shard is a valid training set.",
            "Full train split is 277,650 images / 114 GB.",
        ),
    ),
    DatasetSpec(
        key="coco-val2017",
        name="COCO val2017",
        source="url",
        tier=1,
        role="Real half of the organisers' designated reference benchmark.",
        url="http://images.cocodataset.org/zips/val2017.zip",
        approx_gb=0.82,
        licence="CC BY 4.0 (annotations); per-image Flickr licences",
        homepage="https://cocodataset.org/",
        notes=("5,000 JPEGs, ~640x480. Organisers use 4,998 of them.",),
    ),
    # ---------------- tier 2 : breadth ----------------
    DatasetSpec(
        key="commfor-small",
        name="Community Forensics (small)",
        source="hf_files",
        tier=2,
        role="Generator breadth: 4,803 generators, the strongest driver of unseen-architecture transfer.",
        repo_id="OwensLab/CommunityForensics-Small",
        approx_gb=106.5,
        licence="cc-by-nc-sa-4.0 (NON-COMMERCIAL, viral SA)",
        homepage="https://huggingface.co/datasets/OwensLab/CommunityForensics-Small",
        stat_columns=("format", "resolution", "label", "architecture", "subset"),
        notes=(
            "Shards are SORTED BY LABEL: 0-92 fake, 93 mixed, 94-185 real.",
            "Default selection is stride-3 keeping all of 70-92 (the only GAN/PixDiff data).",
            "Measured confound: 'PNG => fake' alone scores 71.4% balanced accuracy.",
            "Measured confound: every fake is 512x512, 256x256 or 1024x1024.",
            "Contains NO commercial generators (no DALL-E / Midjourney / FLUX).",
        ),
    ),
    DatasetSpec(
        key="dda-coco",
        name="DDA-COCO benchmark",
        source="hf_files",
        tier=2,
        role="Shortcut probe: are we using causal artifacts or dataset bias? Frozen VFMs score <0.08 here.",
        repo_id="Junwei-Xi/DDA-COCO",
        approx_gb=4.3,
        licence="apache-2.0",
        homepage="https://huggingface.co/datasets/Junwei-Xi/DDA-COCO",
        notes=("5 VAE reconstructions of MSCOCO val, frequency-aligned.",),
    ),
    DatasetSpec(
        key="dda-train",
        name="DDA training pairs",
        source="generate",
        tier=2,
        role="Near-boundary hard negatives + the format-alignment protocol.",
        repo_id="Junwei-Xi/DDA-Training-Set",
        approx_gb=113.0,
        licence="apache-2.0",
        homepage="https://huggingface.co/datasets/Junwei-Xi/DDA-Training-Set",
        notes=(
            "DO NOT DOWNLOAD: 11-part split ZIP, unstreamable, ~226 GB peak disk.",
            "Regenerate instead: COCO train2017 + SD-2.1 VAE encode/decode, ~1 h for 20-30k pairs.",
        ),
    ),
    DatasetSpec(
        key="mirage",
        name="MIRAGE",
        source="hf_files",
        tier=2,
        role="In-the-wild evaluation, human-verified. Best value-per-GB of any eval set.",
        repo_id="MIRAGE-GROUP/MIRAGE",
        approx_gb=1.31,
        licence="cc-by-nc-4.0",
        homepage="https://huggingface.co/datasets/MIRAGE-GROUP/MIRAGE",
        stat_columns=("label", "source"),
    ),
    # ---------------- tier 3 : optional ----------------
    DatasetSpec(
        key="sid-set",
        name="SID_Set",
        source="hf_files",
        tier=3,
        role="Only for tampering/localisation. Synthetic half is essentially one generator (FLUX).",
        repo_id="saberzl/SID_Set",
        approx_gb=140.0,
        licence="cc-by-4.0",
        homepage="https://huggingface.co/datasets/saberzl/SID_Set",
        stat_columns=("label",),
        notes=("Worst value-per-GB in the plan; 100k tampered images with masks is its unique asset.",),
    ),
    DatasetSpec(
        key="cifake",
        name="CIFAKE",
        source="kaggle",
        tier=3,
        role="Pipeline smoke test ONLY. 32x32; the six target transformations are meaningless at that size.",
        repo_id="birdy654/cifake-real-and-ai-generated-synthetic-images",
        approx_gb=0.11,
        licence="see Kaggle page",
        homepage="https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images",
        notes=("Never train a deployable detector on this; reals and fakes have different resampling histories.",),
    ),
    DatasetSpec(
        key="wildfake-dalle",
        name="WildFake DALL-E subset",
        source="modelscope",
        tier=3,
        role="Fake half of the organisers' reference benchmark ('DALL-E Advanced' = DALL-E 3).",
        repo_id="hy2628982280/WildFake",
        files=("Images/Diffusion_based/DALLE.zip",),
        approx_gb=25.6,
        licence="apache-2.0 (re-uploader tag)",
        homepage="https://modelscope.cn/datasets/hy2628982280/WildFake/summary",
        notes=(
            "China-hosted; expect slow transfer.",
            "Monolithic 25.6 GB zip holds both DALL-E 2 (Typical) and DALL-E 3 (Advanced).",
        ),
    ),
    DatasetSpec(
        key="synthbuster",
        name="Synthbuster",
        source="url",
        tier=3,
        role="9 commercial generators vs RAISE, uncompressed so we control degradation.",
        url="https://zenodo.org/records/10066460",
        approx_gb=12.4,
        licence="see Zenodo record",
        homepage="https://zenodo.org/records/10066460",
        notes=("Reals (RAISE-1k) are a separate, form-gated download.",),
    ),
)


BY_KEY = {spec.key: spec for spec in REGISTRY}


def select(keys: list[str] | None = None, tiers: list[int] | None = None) -> list[DatasetSpec]:
    chosen = list(REGISTRY)
    if keys:
        chosen = [s for s in chosen if s.key in set(keys)]
    if tiers:
        chosen = [s for s in chosen if s.tier in set(tiers)]
    return chosen


def commfor_shard_selection(stride: int = 3, total: int = 186) -> list[int]:
    """Shard indices for Community Forensics that preserve architecture coverage.

    Shards are sorted by label and by subset, so a naive prefix or uniform
    stride silently drops every GAN and pixel-diffusion image. Shards 70-92
    hold all of the non-latent-diffusion fakes and are always kept in full.
    """
    manual_fakes = list(range(70, 93))
    systematic = list(range(0, 70, stride))
    reals = list(range(94, total, stride))
    return sorted(set(manual_fakes + systematic + reals + [93]))
