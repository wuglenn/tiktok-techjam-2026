"""Report dataset statistics from remote metadata, downloading no images.

Everything here comes from HTTP metadata endpoints -- the HuggingFace repo
tree (exact per-file byte sizes), the datasets-server ``size`` and ``info``
endpoints (row counts and schema), and the ``statistics`` endpoint (per-column
value distributions). For Parquet datasets that store resolution and format as
columns, this is enough to audit real/fake confounds across hundreds of
thousands of images without fetching a single one.

    python dataset_stats.py                  # summary of every dataset
    python dataset_stats.py --tier 1
    python dataset_stats.py --only commfor-small --audit
    python dataset_stats.py --json stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

sys.path.insert(0, "src")

from seer.datasets_registry import DatasetSpec, select  # noqa: E402

HF_API = "https://huggingface.co/api"
DS_SERVER = "https://datasets-server.huggingface.co"
TIMEOUT = 45


def _get(url: str) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            return json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _next_link(header: str | None) -> str | None:
    """Parse an RFC 5988 Link header for rel="next"."""
    if not header:
        return None
    for part in header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        if 'rel="next"' in section[1].replace(" ", "").replace("'", '"'):
            return section[0].strip().strip("<>")
    return None


def _get_all_pages(url: str, max_pages: int = 60) -> list[dict]:
    """Follow Link: rel=next pagination, which the HF tree API uses."""
    items: list[dict] = []
    current: str | None = url
    for _ in range(max_pages):
        if not current:
            break
        try:
            with urllib.request.urlopen(current, timeout=TIMEOUT) as response:
                page = json.load(response)
                link = response.headers.get("Link")
        except Exception:
            break
        if not isinstance(page, list):
            break
        items.extend(page)
        current = _next_link(link)
    return items


def _gb(num_bytes: float | None) -> float | None:
    return None if num_bytes is None else round(num_bytes / 1e9, 2)


def _fmt(value: Any, width: int = 0) -> str:
    text = "?" if value is None else f"{value:,}" if isinstance(value, int) else str(value)
    return text.rjust(width) if width else text


# --------------------------------------------------------------------------
# HuggingFace metadata
# --------------------------------------------------------------------------

def hf_repo_metadata(repo_id: str) -> dict:
    info = _get(f"{HF_API}/datasets/{repo_id}") or {}

    # Plain `recursive=true` returns the whole tree in one response and still
    # carries real file sizes. Adding `expand=true` caps the page at 50 entries
    # and forces Link-header pagination, and combining it with `limit` is a 400.
    # Link following is kept as a fallback in case that behaviour changes.
    tree = _get_all_pages(f"{HF_API}/datasets/{repo_id}/tree/main?recursive=true")
    files: list[tuple[str, int]] = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
        files.append((entry["path"], int(size)))

    card = info.get("cardData") or {}
    return {
        "license": card.get("license") or info.get("license") or "untagged",
        "gated": info.get("gated", False),
        "private": info.get("private", False),
        "downloads_last_month": info.get("downloads"),
        "last_modified": info.get("lastModified"),
        "n_files": len(files),
        "total_bytes": sum(size for _, size in files),
        "files": sorted(files, key=lambda item: -item[1])[:12],
    }


def hf_size_and_schema(repo_id: str) -> dict:
    size = _get(f"{DS_SERVER}/size?dataset={urllib.parse.quote(repo_id)}") or {}
    info = _get(f"{DS_SERVER}/info?dataset={urllib.parse.quote(repo_id)}") or {}

    dataset_size = (size.get("size") or {}).get("dataset") or {}
    splits = [
        {
            "split": s.get("split"),
            "rows": s.get("num_rows"),
            "bytes": s.get("num_bytes_parquet_files"),
        }
        for s in (size.get("size") or {}).get("splits", [])
    ]
    # datasets-server only converts a prefix of very large repos; without this
    # flag its row counts look authoritative when they are a small sample.
    partial = bool(size.get("partial"))

    configs = (info.get("dataset_info") or {})
    features: list[str] = []
    for config in configs.values():
        features = list((config.get("features") or {}).keys())
        break

    return {
        "rows": dataset_size.get("num_rows"),
        "parquet_bytes": dataset_size.get("num_bytes_parquet_files"),
        "partial": partial,
        "splits": splits,
        "columns": features,
    }


def hf_column_statistics(repo_id: str, config: str, split: str) -> dict:
    """Per-column value distributions. This is what makes the confound audit free."""
    query = urllib.parse.urlencode({"dataset": repo_id, "config": config, "split": split})
    payload = _get(f"{DS_SERVER}/statistics?{query}")
    if not payload:
        return {}

    out: dict[str, Any] = {}
    for column in payload.get("statistics", []):
        name = column.get("column_name")
        stats = column.get("column_statistics") or {}
        entry: dict[str, Any] = {"type": column.get("column_type")}
        if "frequencies" in stats:
            freqs = stats["frequencies"]
            entry["top_values"] = sorted(freqs.items(), key=lambda kv: -kv[1])[:12]
            entry["n_unique"] = stats.get("n_unique", len(freqs))
        for key in ("min", "max", "mean", "nan_count", "n_unique"):
            if key in stats and key not in entry:
                entry[key] = stats[key]
        out[name] = entry
    return out


# --------------------------------------------------------------------------
# Non-HuggingFace sources
# --------------------------------------------------------------------------

def head_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def modelscope_files(repo_id: str) -> list[tuple[str, int]]:
    url = f"https://modelscope.cn/api/v1/datasets/{repo_id}/repo/tree?Revision=master&Root=/"
    payload = _get(url) or {}
    files = ((payload.get("Data") or {}).get("Files")) or []
    return [(f.get("Path", "?"), int(f.get("Size") or 0)) for f in files]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def describe(spec: DatasetSpec, audit: bool) -> dict:
    record: dict[str, Any] = {
        "key": spec.key,
        "name": spec.name,
        "tier": spec.tier,
        "source": spec.source,
        "role": spec.role,
        "planned_gb": spec.approx_gb,
        "licence_expected": spec.licence,
        "homepage": spec.homepage,
        "notes": list(spec.notes),
    }

    if spec.source in ("hf", "hf_files", "generate") and spec.repo_id:
        meta = hf_repo_metadata(spec.repo_id)
        record["licence_actual"] = meta["license"]
        record["gated"] = meta["gated"]
        record["downloads_last_month"] = meta["downloads_last_month"]
        record["n_files"] = meta["n_files"]
        # Exact, from the repo tree. Authoritative: prefer over datasets-server.
        record["repo_files_gb"] = _gb(meta["total_bytes"])
        record["largest_files"] = [(path, _gb(size)) for path, size in meta["files"]]

        schema = hf_size_and_schema(spec.repo_id)
        record["rows"] = schema["rows"]
        record["rows_partial"] = schema["partial"]
        record["parquet_gb"] = _gb(schema["parquet_bytes"])
        record["columns"] = schema["columns"]
        record["splits"] = schema["splits"]

        if audit and spec.stat_columns:
            stats = hf_column_statistics(spec.repo_id, spec.config, spec.split)
            record["column_stats"] = {k: v for k, v in stats.items() if k in spec.stat_columns}
            if not record["column_stats"]:
                # Silence here would be dangerous: for partially-indexed repos
                # the server's sample is the first shards, which for
                # CommunityForensics are 100% one class. Any statistic derived
                # from it would look authoritative and be badly wrong.
                record["audit_unavailable"] = (
                    "datasets-server has no column statistics for this repo"
                    + (
                        f" (only {record['rows']:,} of the full set indexed, and the indexed"
                        " prefix is not class-balanced -- do not infer distributions from it)"
                        if record.get("rows_partial")
                        else ""
                    )
                )

    elif spec.source == "url" and spec.url.endswith(".zip"):
        record["remote_gb"] = _gb(head_size(spec.url))

    elif spec.source == "modelscope":
        wanted = set(spec.files)
        files = modelscope_files(spec.repo_id)
        record["remote_files"] = [(p, _gb(s)) for p, s in files if not wanted or p in wanted][:12]

    return record


def resolved_size_gb(record: dict) -> float | None:
    """Repo tree bytes are exact; datasets-server is a fallback and may be partial."""
    for key in ("repo_files_gb", "remote_gb", "parquet_gb"):
        value = record.get(key)
        if value:
            return float(value)
    return float(record["planned_gb"]) or None


def print_summary(records: list[dict]) -> None:
    header = f"{'key':<16} {'tier':>4} {'files':>7} {'rows':>10} {'size GB':>9}  {'gated':<6} licence"
    print(header)
    print("-" * len(header))
    total = 0.0
    for record in records:
        size = resolved_size_gb(record)
        total += size or 0.0
        rows = _fmt(record.get("rows"), 10)
        if record.get("rows_partial"):
            rows = rows.strip() + "+"
            rows = rows.rjust(10)
        print(
            f"{record['key']:<16} {record['tier']:>4} {_fmt(record.get('n_files'), 7)} {rows} "
            f"{(f'{size:.2f}' if size else '?'):>9}  "
            f"{str(record.get('gated', '-')):<6} {record.get('licence_actual', record['licence_expected'])}"
        )
    print("-" * len(header))
    print(f"{'TOTAL':<16} {'':>4} {'':>7} {'':>10} {total:>9.2f}")
    if any(r.get("rows_partial") for r in records):
        print("\n  '+' = datasets-server only indexed a prefix; true row count is higher.")


def print_detail(record: dict) -> None:
    print(f"\n=== {record['name']}  [{record['key']}, tier {record['tier']}] ===")
    print(f"  role      : {record['role']}")
    print(f"  homepage  : {record['homepage']}")
    print(f"  licence   : {record.get('licence_actual', record['licence_expected'])}"
          f"   gated={record.get('gated', '-')}")
    if record.get("rows"):
        print(f"  rows      : {record['rows']:,}")
    for key in ("parquet_gb", "repo_files_gb", "remote_gb"):
        if record.get(key):
            print(f"  {key:<10}: {record[key]} GB")
    if record.get("splits"):
        for split in record["splits"]:
            print(f"    split {split['split']:<12} rows={_fmt(split['rows'])}  bytes={_gb(split['bytes'])} GB")
    if record.get("columns"):
        print(f"  columns   : {', '.join(record['columns'])}")
    if record.get("largest_files"):
        print("  largest files:")
        for path, size in record["largest_files"][:6]:
            print(f"    {size if size else '?':>7} GB  {path}")
    if record.get("remote_files"):
        print("  remote files:")
        for path, size in record["remote_files"]:
            print(f"    {size if size else '?':>7} GB  {path}")

    for column, stats in (record.get("column_stats") or {}).items():
        print(f"  [{column}] type={stats.get('type')} unique={stats.get('n_unique', '?')}")
        for value, count in stats.get("top_values", [])[:8]:
            print(f"      {count:>10,}  {value}")
    if record.get("audit_unavailable"):
        print(f"  audit     : {record['audit_unavailable']}")

    for note in record["notes"]:
        print(f"  ! {note}")


def scan_local() -> list[dict]:
    """Class balance of whatever is already on disk, read from label CSVs only.

    Label conventions differ between datasets and getting one wrong silently
    inverts a whole training run, so the convention is reported alongside the
    counts rather than assumed.
    """
    import csv as _csv

    from seer.paths import ntire_root

    root = ntire_root()
    targets = [
        ("ntire-train shard_0", root / "NTIRE-RobustAIGenDetection-train" / "shard_0" / "shard_0" / "labels.csv"),
        ("ntire-val", root / "NTIRE-RobustAIGenDetection-val" / "val_labels.csv"),
        ("ntire-val-hard", root / "NTIRE-RobustAIGenDetection-val" / "val_hard_labels.csv"),
        ("ntire-test", root / "NTIRE-RobustAIGenDetection-test-public" / "test_labels.csv"),
    ]

    records = []
    for name, path in targets:
        if not path.exists():
            continue
        real = fake = distorted = 0
        distortion_counts: dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in _csv.DictReader(handle):
                label = (row.get("label") or "").strip()
                if label == "0":
                    real += 1
                elif label == "1":
                    fake += 1
                if (row.get("is_distorted") or "0").strip() == "1":
                    distorted += 1
                    raw = row.get("distortions") or ""
                    for token in raw.strip("[]").replace("'", "").split(","):
                        token = token.strip()
                        if token:
                            distortion_counts[token] = distortion_counts.get(token, 0) + 1
        total = real + fake
        records.append({
            "name": name,
            "path": str(path),
            "total": total,
            "real": real,
            "fake": fake,
            "fake_pct": round(100.0 * fake / total, 1) if total else 0.0,
            "distorted": distorted,
            "distortion_counts": dict(sorted(distortion_counts.items(), key=lambda kv: -kv[1])),
        })
    return records


def print_local(records: list[dict]) -> None:
    if not records:
        print("nothing downloaded yet -- run: python get_datasets.py --tier 1")
        return
    print("\nOn disk (label convention: 0 = real / authentic, 1 = AI-generated)\n")
    header = f"{'split':<22} {'total':>8} {'real':>8} {'fake':>8} {'fake %':>7} {'degraded':>9}"
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{record['name']:<22} {record['total']:>8,} {record['real']:>8,} "
            f"{record['fake']:>8,} {record['fake_pct']:>6.1f}% {record['distorted']:>9,}"
        )
    for record in records:
        if record["distortion_counts"]:
            print(f"\n  {record['name']} degradation vocabulary (applied by the organisers):")
            for token, count in list(record["distortion_counts"].items()):
                print(f"    {count:>6,}  {token}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="dataset keys")
    parser.add_argument("--tier", type=int, nargs="*", help="tiers to include")
    parser.add_argument("--audit", action="store_true", help="fetch per-column value distributions")
    parser.add_argument("--detail", action="store_true", help="print full per-dataset detail")
    parser.add_argument("--local", action="store_true", help="class balance of datasets already on disk")
    parser.add_argument("--json", type=str, help="also write raw records to this path")
    args = parser.parse_args()

    if args.local:
        print_local(scan_local())
        return

    specs = select(args.only, args.tier)
    if not specs:
        print("no datasets matched")
        return

    print(f"inspecting {len(specs)} dataset(s) via remote metadata only -- no images downloaded\n")
    records = []
    for spec in specs:
        print(f"  querying {spec.key} ...", end="\r", flush=True)
        records.append(describe(spec, args.audit))
    print(" " * 40, end="\r")

    print_summary(records)
    if args.detail or args.audit:
        for record in records:
            print_detail(record)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
