"""Map dataset-specific label conventions onto ours: 0 = real, 1 = fake.

Several public sets invert this (or use strings). The ClassLabel on
``julienlucas/midjourney-dalle-sd-nanobananapro-dataset`` is inverted
(0 = fake, 1 = real). ``saberzl/SID_Set`` is three-class (0 = real,
1 = full synthetic, 2 = tampered) — map 1 and 2 to fake.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# String aliases when the column is text rather than a ClassLabel id.
_FAKE = {"fake", "ai", "generated", "synthetic", "ai-generated", "ai_generated"}
_REAL = {"real", "authentic", "human", "photo"}


def normalize_label(
    raw: Any,
    label_map: Optional[Mapping[Any, int]] = None,
    default: Optional[int] = None,
) -> Optional[int]:
    """Return 0 (real) or 1 (fake), or ``default`` when the value is empty."""
    if raw is None or raw == "":
        return default
    if label_map:
        mapped = _lookup(label_map, raw)
        if mapped is not None:
            return int(mapped)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _FAKE:
            return 1
        if key in _REAL:
            return 0
        if key in ("1", "true", "yes"):
            return 1
        if key in ("0", "false", "no"):
            return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _lookup(label_map: Mapping[Any, int], raw: Any) -> Optional[int]:
    if raw in label_map:
        return label_map[raw]
    # YAML keys are often strings; HF ClassLabel ids are ints / numpy ints.
    as_str = str(raw).strip()
    if as_str in label_map:
        return label_map[as_str]
    try:
        as_int = int(raw)
    except (TypeError, ValueError):
        return None
    if as_int in label_map:
        return label_map[as_int]
    if str(as_int) in label_map:
        return label_map[str(as_int)]
    return None
