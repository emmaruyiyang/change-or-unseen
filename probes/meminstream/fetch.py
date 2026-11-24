"""Fetch a stratified sample of MemInStream (HF zizi32/MemInStream) without downloading the 310 GB of zips.

1. Read the central directory of each videos_XXX.zip by HTTP range request (cached in data/meminstream/zip_listing.json).
2. Pick N questions per level-2 task (17 tasks), preferring small videos and videos that carry several sampled
   questions, all with a query time >= MIN_T s.
3. Range-read just those zip members into data/meminstream/videos/.
Writes data/meminstream/sample.jsonl (the sampled questions; `local` = the local video file).

Usage: .venv/bin/python probes/meminstream/fetch.py --per-task 3
"""
import argparse, collections, io, json, os, sys, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(ROOT, "methods", "stamem"))
from fetch_long import HttpFile  # noqa: E402  (range-request file object, certifi SSL)

DATA = os.path.join(ROOT, "data", "meminstream")
URL = "https://huggingface.co/datasets/zizi32/MemInStream/resolve/main/videos_{:03d}.zip"
MIN_T = 120


def listing():
    path = os.path.join(DATA, "zip_listing.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for c in range(1, 11):
        if str(c) in out:
            continue
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL.format(c)), buffer_size=1 << 20))
        out[str(c)] = [(i.filename, i.file_size, i.compress_type) for i in z.infolist() if not i.is_dir()]
        json.dump(out, open(path, "w"))
        print(f"zip {c:03d}: {len(out[str(c)])} files", flush=True)
    return {os.path.basename(n): (int(c), n, s) for c, items in out.items() for n, s, _ in items}


def main(args):
    ann = json.load(open(os.path.join(DATA, "annotations.json")))
    where = listing()
    qs = [q for q in ann if q["query_time"] >= MIN_T and os.path.basename(q["video_path"]) in where]
    size = lambda q: where[os.path.basename(q["video_path"])][2]
    per_video = collections.Counter(q["video_path"] for q in qs)
    pick, chosen_videos = [], set()
    by_task = collections.defaultdict(list)
    for q in qs:
        by_task[q["level2"]].append(q)
    for task in sorted(by_task):
        # prefer videos already chosen, then small videos with many questions
        cand = sorted(by_task[task], key=lambda q: (q["video_path"] not in chosen_videos,
                                                    size(q) / per_video[q["video_path"]]))
        cand = [q for q in cand if size(q) < args.max_mb * 1e6][:args.per_task]
        pick += cand
        chosen_videos |= {q["video_path"] for q in cand}
    total = sum(where[os.path.basename(v)][2] for v in chosen_videos)
    print(f"{len(pick)} questions, {len(chosen_videos)} videos, {total / 1e9:.2f} GB", flush=True)
    if args.dry_run:
        return
    os.makedirs(os.path.join(DATA, "videos"), exist_ok=True)
    for v in sorted(chosen_videos):
        c, name, sz = where[os.path.basename(v)]
        out = os.path.join(DATA, "videos", os.path.basename(v))
        if os.path.exists(out) and os.path.getsize(out) == sz:
            continue
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(URL.format(c)), buffer_size=1 << 20))
        with z.open(name) as src, open(out + ".part", "wb") as fo:
            while True:
                b = src.read(32 << 20)
                if not b:
                    break
                fo.write(b)
        os.replace(out + ".part", out)
        print(f"{os.path.basename(v)[:60]} {sz / 1e6:.0f} MB", flush=True)
    with open(os.path.join(DATA, "sample.jsonl"), "w") as fo:
        for q in pick:
            fo.write(json.dumps({**q, "local": os.path.basename(q["video_path"])}) + "\n")
    print(f"-> data/meminstream/sample.jsonl ({len(pick)} questions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--max-mb", type=float, default=300)
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
