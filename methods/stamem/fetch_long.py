"""Fetch single long videos for the StaMem headroom check, without downloading whole archives.

  ovo      OVO-Bench backward questions (EPM/ASI/HLD) asked at t >= 120 s. Videos are cut out of
           HF JoeLeelyf/OVO-Bench src_videos.tar.part* by byte range (header index: data/ovo/src_videos_index.jsonl).
           Picks the N videos with the most such questions per MB.
  videomme Video-MME "long" videos (30-60 min). Read the central directory of each
           HF lmms-eval/Video-MME videos_chunked_*.zip by range request, then range-read single members.
           Picks N long videos, spread over domains.

  ovo_extra adds n videos each for ASI and HLD (asked at t >= 300 s) -> data/ovo/long_backward.jsonl

Usage: .venv/bin/python methods/stamem/fetch_long.py ovo --n 20
       .venv/bin/python methods/stamem/fetch_long.py ovo_extra --n 10
       .venv/bin/python methods/stamem/fetch_long.py videomme --n 20
"""
import argparse, collections, io, json, os, ssl, sys, urllib.request, zipfile

import certifi

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
OVO_URL = "https://huggingface.co/datasets/JoeLeelyf/OVO-Bench/resolve/main/src_videos.tar.part{}"
OVO_PARTS = ["aa", "ab", "ac", "ad", "ae"]
VMME_URL = "https://huggingface.co/datasets/lmms-eval/Video-MME/resolve/main/videos_chunked_{:02d}.zip"
VMME_PARQUET = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--lmms-eval--Video-MME/snapshots/"
    "ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet")
UA = {"User-Agent": "research-script"}
CTX = ssl.create_default_context(cafile=certifi.where())


def get_range(url, a, b):
    """bytes [a, b) of url."""
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
        return r.read()


def size_of(url):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return int(r.headers["Content-Length"])


def copy_range(url, a, b, fo, chunk=32 << 20):
    while a < b:
        e = min(a + chunk, b)
        fo.write(get_range(url, a, e))
        a = e


# ---------------- OVO ----------------
def ovo(n):
    d = json.load(open(os.path.join(ROOT, "ovo", "ovo_bench_new.json")))
    idx = {}
    for l in open(os.path.join(ROOT, "ovo", "src_videos_index.jsonl")):
        r = json.loads(l)
        idx[r["name"].replace("./src_videos/", "")] = r
    qs = [x for x in d if x["task"] in ("EPM", "ASI", "HLD") and x["realtime"] >= 120]
    cnt = collections.Counter(x["video"] for x in qs)
    vids = sorted(cnt, key=lambda v: -cnt[v] / max(idx[v]["size"] / 1e6, 1))
    vids = [v for v in vids if idx[v]["size"] < 400e6][:n]
    sizes = [size_of(OVO_URL.format(p)) for p in OVO_PARTS]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    for v in vids:
        out = os.path.join(ROOT, "ovo", "videos", v)
        if os.path.exists(out) and os.path.getsize(out) == idx[v]["size"]:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        a, b = idx[v]["offset"], idx[v]["offset"] + idx[v]["size"]
        with open(out + ".part", "wb") as fo:
            for p, s0, sz in zip(OVO_PARTS, starts, sizes):  # a video may straddle two parts
                lo, hi = max(a, s0), min(b, s0 + sz)
                if lo < hi:
                    copy_range(OVO_URL.format(p), lo - s0, hi - s0, fo)
        os.replace(out + ".part", out)
        print(f"ovo {v} {idx[v]['size'] / 1e6:.0f} MB, {cnt[v]} questions", flush=True)
    sel = [x for x in qs if x["video"] in set(vids)]
    with open(os.path.join(ROOT, "ovo", "long_samples.jsonl"), "w") as fo:
        for x in sel:
            fo.write(json.dumps({**x, "local": x["video"]}) + "\n")
    print(f"ovo: {len(vids)} videos, {len(sel)} questions -> data/ovo/long_samples.jsonl")


def ovo_extra(n, tasks=("ASI", "HLD"), min_t=300):
    """Add n videos per task (questions asked at t >= min_t) for the backward P0-P3 probe, then write
    data/ovo/long_backward.jsonl: every backward question at t >= 120 s whose video is local."""
    d = json.load(open(os.path.join(ROOT, "ovo", "ovo_bench_new.json")))
    idx = {}
    for l in open(os.path.join(ROOT, "ovo", "src_videos_index.jsonl")):
        r = json.loads(l)
        idx[r["name"].replace("./src_videos/", "")] = r
    local = lambda v: os.path.exists(os.path.join(ROOT, "ovo", "videos", v))
    sizes = [size_of(OVO_URL.format(p)) for p in OVO_PARTS]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    for task in tasks:
        cnt = collections.Counter(x["video"] for x in d
                                  if x["task"] == task and x["realtime"] >= min_t and not local(x["video"]))
        vids = [v for v in sorted(cnt, key=lambda v: -cnt[v] / idx[v]["size"]) if idx[v]["size"] < 300e6][:n]
        for v in vids:
            out = os.path.join(ROOT, "ovo", "videos", v)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            a, b = idx[v]["offset"], idx[v]["offset"] + idx[v]["size"]
            with open(out + ".part", "wb") as fo:
                for p, s0, sz in zip(OVO_PARTS, starts, sizes):
                    lo, hi = max(a, s0), min(b, s0 + sz)
                    if lo < hi:
                        copy_range(OVO_URL.format(p), lo - s0, hi - s0, fo)
            os.replace(out + ".part", out)
            print(f"ovo {task} {v} {idx[v]['size'] / 1e6:.0f} MB, {cnt[v]} questions", flush=True)
    sel = [x for x in d if x["task"] in ("EPM", "ASI", "HLD") and x["realtime"] >= 120 and local(x["video"])]
    with open(os.path.join(ROOT, "ovo", "long_backward.jsonl"), "w") as fo:
        for x in sel:
            fo.write(json.dumps({**x, "local": x["video"]}) + "\n")
    print(f"ovo backward: {len(sel)} questions {dict(collections.Counter(x['task'] for x in sel))} "
          "-> data/ovo/long_backward.jsonl")


# ---------------- Video-MME ----------------
class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        self.url, self.pos, self.size = url, 0, size_of(url)

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, k=-1):
        if k is None or k < 0:
            k = self.size - self.pos
        k = min(k, self.size - self.pos)
        if k <= 0:
            return b""
        data = get_range(self.url, self.pos, self.pos + k)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def videomme(n):
    import pandas as pd
    df = pd.read_parquet(VMME_PARQUET)
    long = df[df.duration == "long"]
    want = set(long.videoID)
    dom = dict(zip(long.videoID, long.domain))
    out_dir = os.path.join(ROOT, "videomme", "videos")
    os.makedirs(out_dir, exist_ok=True)
    listing_path = os.path.join(ROOT, "videomme", "zip_listing.json")
    listing = json.load(open(listing_path)) if os.path.exists(listing_path) else {}
    for c in range(1, 21):
        if str(c) in listing:
            continue
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(VMME_URL.format(c)), buffer_size=1 << 20))
        listing[str(c)] = [(i.filename, i.file_size) for i in z.infolist()
                           if os.path.splitext(os.path.basename(i.filename))[0] in want]
        json.dump(listing, open(listing_path, "w"))
        print(f"zip {c:02d}: {len(listing[str(c)])} long videos", flush=True)
    # spread over domains: round-robin, smallest file first within a domain
    by_dom = collections.defaultdict(list)
    for c, items in listing.items():
        for name, sz in items:
            vid = os.path.splitext(os.path.basename(name))[0]
            by_dom[dom[vid]].append((sz, vid, name, int(c)))
    for v in by_dom.values():
        v.sort()
    pick = []
    while len(pick) < n and any(by_dom.values()):
        for k in sorted(by_dom):
            if by_dom[k] and len(pick) < n:
                pick.append(by_dom[k].pop(0))
    print(f"picked {len(pick)} videos, {sum(p[0] for p in pick) / 1e9:.2f} GB", flush=True)
    for sz, vid, name, c in pick:
        out = os.path.join(out_dir, vid + ".mp4")
        if os.path.exists(out) and os.path.getsize(out) == sz:
            continue
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(VMME_URL.format(c)), buffer_size=1 << 20))
        with z.open(name) as src, open(out + ".part", "wb") as fo:
            while True:
                b = src.read(32 << 20)
                if not b:
                    break
                fo.write(b)
        os.replace(out + ".part", out)
        print(f"videomme {vid} {sz / 1e6:.0f} MB ({dom[vid]})", flush=True)
    sel = long[long.videoID.isin({p[1] for p in pick})]
    sel.to_json(os.path.join(ROOT, "videomme", "long_samples.jsonl"), orient="records", lines=True)
    print(f"videomme: {len(pick)} videos, {len(sel)} questions -> data/videomme/long_samples.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["ovo", "ovo_extra", "videomme"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    {"ovo": ovo, "ovo_extra": ovo_extra, "videomme": videomme}[a.which](a.n)
