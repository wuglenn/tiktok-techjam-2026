"""Probe Community Forensics parquet shards over HTTP range requests.

Reads only parquet footers (no image bytes) to recover exact row counts and
per-column-chunk statistics for label / format / model_name / resolution.
"""
import io
import json
import sys
import concurrent.futures as cf

import requests
import pyarrow.parquet as pq

SESSION = requests.Session()


class HttpFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        headers = {"Range": f"bytes={self.pos}-{end}"}
        r = SESSION.get(self.url, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data


def list_files(repo, subdir_filter=None):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/data?recursive=true&expand=true"
    out = []
    cursor = None
    while True:
        u = url + (f"&cursor={cursor}" if cursor else "")
        r = SESSION.get(u, timeout=120)
        r.raise_for_status()
        items = r.json()
        for it in items:
            if it.get("type") == "file":
                size = it.get("size") or (it.get("lfs") or {}).get("size")
                out.append((it["path"], size))
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            cursor = link.split("cursor=")[1].split(">")[0].split("&")[0]
        else:
            break
    if subdir_filter:
        out = [x for x in out if x[0].startswith(subdir_filter)]
    return out


def probe(repo, path, size):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    f = HttpFile(url, size)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = [md.schema.column(i).name for i in range(md.num_columns)]
    want = {n: names.index(n) for n in ("label", "format", "model_name", "resolution", "subset", "architecture") if n in names}
    stats = {k: {} for k in want}
    label_rows = {}
    fmt_rows = {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        n = g.num_rows
        for col, idx in want.items():
            st = g.column(idx).statistics
            if st is None or not st.has_min_max:
                key = "<nostats>"
            else:
                mn = st.min if isinstance(st.min, str) else str(st.min)
                mx = st.max if isinstance(st.max, str) else str(st.max)
                key = mn if mn == mx else f"{mn}..{mx}"
            stats[col][key] = stats[col].get(key, 0) + n
        if "label" in want:
            st = g.column(want["label"]).statistics
            if st is not None and st.has_min_max and st.min == st.max:
                label_rows[st.min] = label_rows.get(st.min, 0) + n
            else:
                label_rows["MIXED"] = label_rows.get("MIXED", 0) + n
        if "format" in want:
            st = g.column(want["format"]).statistics
            if st is not None and st.has_min_max and st.min == st.max:
                fmt_rows[st.min] = fmt_rows.get(st.min, 0) + n
            else:
                fmt_rows["MIXED"] = fmt_rows.get("MIXED", 0) + n
    return {
        "path": path,
        "bytes": size,
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "label_rows": label_rows,
        "fmt_rows": fmt_rows,
        "stats": stats,
        "columns": names,
    }


def run(repo, out_json):
    files = [x for x in list_files(repo) if x[0].endswith(".parquet")]
    print(f"{repo}: {len(files)} parquet files, total bytes = {sum(s for _, s in files):,}")
    results = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(probe, repo, p, s): p for p, s in files}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                results.append(fu.result())
            except Exception as e:
                print("FAIL", futs[fu], repr(e))
            if (i + 1) % 25 == 0:
                print(f"  ...{i+1}/{len(files)}", flush=True)
    total_rows = sum(r["num_rows"] for r in results)
    lab = {}
    fmt = {}
    for r in results:
        for k, v in r["label_rows"].items():
            lab[k] = lab.get(k, 0) + v
        for k, v in r["fmt_rows"].items():
            fmt[k] = fmt.get(k, 0) + v
    print(f"TOTAL rows = {total_rows:,}")
    print(f"label (row-group-pure) = {lab}")
    print(f"format (row-group-pure) = {fmt}")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
