"""Probe the HuggingFace tree API's pagination behaviour.

Kept because the correct parameter combination is not documented clearly and
the failure mode is silent (an empty list rather than an error).
"""

import json
import urllib.request

BASE = "https://huggingface.co/api/datasets/OwensLab/CommunityForensics-Small/tree/main"

QUERIES = [
    "?recursive=true&expand=true&limit=1000",
    "?recursive=true&expand=true",
    "?recursive=true&limit=1000",
    "?recursive=true",
    "",
]

for query in QUERIES:
    try:
        response = urllib.request.urlopen(BASE + query, timeout=45)
        payload = json.load(response)
        link = response.headers.get("Link")
        print(f"{query or '(none)':<42} -> {len(payload):>4} entries   link={link!r}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"{query or '(none)':<42} -> ERROR {type(exc).__name__}: {exc}")
