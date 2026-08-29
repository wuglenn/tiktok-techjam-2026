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

Source = Literal["hf", "hf_files", "url", "kaggle", "modelscope", "generate", "manual", "stream"]


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
        name="NTIRE 2026 train (all 6 shards)",
        source="hf_files",
        tier=1,
        role="Primary training source: 42 generators (2022-2026), real/fake matched on resolution, aspect ratio and JPEG quality.",
        repo_id=NTIRE_TRAIN,
        files=tuple(f"shard_{i}.zip" for i in range(6)),
        approx_gb=114.0,
        licence="untagged",
        homepage=f"https://huggingface.co/datasets/{NTIRE_TRAIN}",
        notes=(
            "Full train split: 277,650 images across 6 distribution-matched shards.",
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
        approx_gb=260.0,
        licence="cc-by-nc-sa-4.0 (NON-COMMERCIAL, viral SA)",
        homepage="https://huggingface.co/datasets/OwensLab/CommunityForensics-Small",
        stat_columns=("format", "resolution", "label", "architecture", "subset"),
        notes=(
            "Full Small dump: ~186 parquet shards, ~260 GB.",
            "Shards are SORTED BY LABEL: 0-92 fake, 93 mixed, 94-185 real.",
            "Measured confound: 'PNG => fake' alone scores 71.4% balanced accuracy.",
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
        name="DDA training pairs (fakes)",
        source="hf_files",
        tier=2,
        role="Fake-only: DDA-aligned VAE reconstructions of COCO train. Near-boundary hard negatives.",
        repo_id="Junwei-Xi/DDA-Training-Set",
        files=(
            "DDA-Training-Set_split.zip",
            *(f"DDA-Training-Set_split.z{i:02d}" for i in range(1, 11)),
        ),
        approx_gb=113.0,
        licence="apache-2.0",
        homepage="https://huggingface.co/datasets/Junwei-Xi/DDA-Training-Set",
        notes=(
            "11-part split ZIP, not streamable. Peak disk ~226 GB to join+extract.",
            "After fetch, join volumes then organize into fake/ (and real/).",
            "Mixture uses the fake half only (folders source).",
        ),
    ),
    DatasetSpec(
        key="flux-reason-6m",
        name="FLUX-Reason-6M",
        source="stream",
        tier=2,
        role="Fake-only: 5.9M FLUX.1-dev images. Stream; do not snapshot (~882 GB).",
        repo_id="LucasFang/FLUX-Reason-6M",
        approx_gb=882.0,
        licence="apache-2.0",
        homepage="https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M",
        stat_columns=("image",),
        notes=(
            "100% AI-generated (FLUX.1-dev). Treat every row as fake (label=1).",
            "Full dump is 1,180 shards / ~882 GB. Default fetch is 64 shards (~48 GB).",
        ),
    ),
    DatasetSpec(
        key="frontier-fakes",
        name="Midjourney / DALL-E / SD / Nano Banana Pro (fakes)",
        source="hf_files",
        tier=2,
        role="Fake-only slice of a labelled frontier-generator set (Nano Banana Pro, Midjourney, DALL-E, SD).",
        repo_id="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        files=tuple(f"data/train-{i:05d}-of-00009.parquet" for i in range(9)),
        approx_gb=3.1,
        licence="mit",
        homepage="https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        stat_columns=("label",),
        notes=(
            "ClassLabel is INVERTED vs our convention: 0 = fake, 1 = real.",
            "Training keeps only the fake class after remapping (keep_label=1).",
            "Train split is ~10.7k rows, about half fake; test (2k) is held out.",
        ),
    ),
    DatasetSpec(
        key="sid-set",
        name="SID_Set (fakes)",
        source="stream",
        tier=2,
        role="Fake-only: full-synthetic + tampered social-media images (drop the real class).",
        repo_id="saberzl/SID_Set",
        approx_gb=140.0,
        licence="cc-by-4.0",
        homepage="https://huggingface.co/datasets/saberzl/SID_Set",
        stat_columns=("label",),
        notes=(
            "Three classes: 0=real, 1=full synthetic, 2=tampered. Map 1 and 2 → fake.",
            "STREAM: 249 train parquet shards, ~140 GB. Mixture type hf, keep_label=1.",
            "Optional slice: uv run scripts/fetch_data.py sid-set --max-shards 16",
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
