"""MemInStream P2 probe (answer from the annotated evidence frames only), self-contained.

Same logic as the run reported in the doc (probes/ovo/ovo_long_probe.py --data meminstream), pulled into one file:
  data      data/meminstream/sample.jsonl  (made by probes/meminstream/fetch.py; videos in data/meminstream/videos/)
  frames    1 fps, one ffmpeg pass per video, long side 448 px -> data/meminstream/frames1fps/<video>/t%05.1f.jpg
  evidence  every 1 fps frame at t <= query_time inside an annotated span (+-0.5 s); a span with no frame inside
            takes the frame nearest its middle; at most 8 frames, spread uniformly
  prompt    "Frame at t=Xs:" + image per frame, then the stem below; options shuffled by a fixed per-question hash
  model     Qwen3-VL-8B-Instruct, MLX 4-bit, greedy, max 8 new tokens
Questions with several correct options (37 of 3,799; 1 in our sample) are skipped.

Usage:
  .venv/bin/python probes/meminstream/p2_probe.py --dry-run        # print frames / gold per question, no model
  .venv/bin/python probes/meminstream/p2_probe.py                  # run, resumable -> probes/meminstream/results/p2_<tag>.jsonl
"""
import argparse, glob, hashlib, json, math, os, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
DATA = os.path.join(ROOT, "data", "meminstream")
MAX_SIDE, MAX_EV = 448, 8
LETTERS = "ABCDE"
MCQ_TAIL = "Answer with the letter of the correct option only."


# ---------- frames ----------
def duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def frames_1fps(video):
    """Extract every 1 fps frame once; return (sorted times, frame-path function)."""
    d = os.path.join(DATA, "frames1fps", os.path.splitext(os.path.basename(video))[0])
    n = int(math.floor(duration(video) - 0.05)) + 1
    if not os.path.exists(os.path.join(d, ".done")) and len(glob.glob(os.path.join(d, "t*.jpg"))) < n:
        os.makedirs(d, exist_ok=True)
        vf = f"fps=1:round=down,scale='if(gt(iw,ih),{MAX_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_SIDE})'"
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vf", vf, "-q:v", "4", "-frames:v", str(n),
                        "-start_number", "0", "-y", os.path.join(d, "f%05d.jpg")], check=True)
        for f in glob.glob(os.path.join(d, "f*.jpg")):
            k = int(re.search(r"f(\d+)\.jpg", f).group(1))
            os.replace(f, os.path.join(d, f"t{float(k):05.1f}.jpg"))
        open(os.path.join(d, ".done"), "w").close()
    times = sorted(float(os.path.basename(f)[1:-4]) for f in glob.glob(os.path.join(d, "t*.jpg")))
    return [t for t in times if t < n], (lambda t: os.path.join(d, f"t{t:05.1f}.jpg"))


# ---------- evidence frames ----------
def spread(times, upto, n):
    ts = [t for t in times if t <= upto]
    if len(ts) <= n:
        return ts
    return [ts[round(i * (len(ts) - 1) / (n - 1))] for i in range(n)]


def evidence_times(times, query_time, spans):
    grid = [t for t in times if t <= query_time]
    pick = set()
    for a, b in spans:
        inside = [t for t in grid if a - 0.5 <= t <= b + 0.5]
        pick.update(inside or [min(grid, key=lambda t: abs(t - (a + b) / 2))])
    return spread(sorted(pick), query_time, MAX_EV)


# ---------- prompt ----------
def qid(r):
    return "mis" + hashlib.md5((r["video_id"] + r["question"]).encode()).hexdigest()[:10]


def permutation(q_id, n):
    """Fixed option order per question (same hash as probes/storm/probe.py)."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{q_id}:{i}".encode()).hexdigest())


def build(r):
    opts = [re.sub(r"^[A-E]\.\s*", "", o) for o in r["options"]]
    perm = permutation(qid(r), len(opts))
    gold = LETTERS[perm.index(r["correct_index"])]
    block = "\n".join(f"{LETTERS[j]}. {opts[i]}" for j, i in enumerate(perm))
    text = (f"These frames are selected moments from a video stream. The current time is t={float(r['query_time']):g}s.\n"
            f"Question: {r['question']}\nOptions:\n{block}\n{MCQ_TAIL}")
    return gold, text


def parse_letter(s):
    m = re.search(r"\b([A-E])\b", s.strip().upper())
    return m.group(1) if m else None


# ---------- model ----------
class VLM:
    def __init__(self, model_id):
        from mlx_vlm import load
        self.model, self.processor = load(model_id)

    def ask(self, frames, times, text, max_tokens=8):
        from mlx_vlm import generate
        import mlx.core as mx
        content = []
        for t in times:
            content += [{"type": "text", "text": f"Frame at t={t:.0f}s:"}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
        res = generate(self.model, self.processor, prompt, image=frames or None,
                       max_tokens=max_tokens, temperature=0.0, verbose=False)
        mx.clear_cache()  # keeps the Metal cache from growing across calls
        return res.text if hasattr(res, "text") else str(res)


def main(a):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "sample.jsonl"))]
    rows = [r for r in rows if isinstance(r["correct_index"], int)]
    out = os.path.join(HERE, "results", f"p2_{a.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    vlm = None if a.dry_run else VLM(a.model)
    cache = {}
    with open(out, "a") as fo:
        for n, r in enumerate(rows):
            if qid(r) in done and not a.dry_run:
                continue
            video = os.path.join(DATA, "videos", r["local"])
            times, frame = cache.setdefault(video, frames_1fps(video))
            ev = evidence_times(times, float(r["query_time"]), r["evidence"])
            gold, text = build(r)
            if a.dry_run:
                print(f"{qid(r)} gold={gold} t={r['query_time']} spans={r['evidence']} frames={ev}")
                continue
            t0 = time.time()
            raw = vlm.ask([frame(t) for t in ev], ev, text)
            pred = parse_letter(raw)
            rec = {"id": qid(r), "level1": r["level1"], "type": r["level2"], "query_time": r["query_time"],
                   "evidence": r["evidence"], "frames": ev, "gold": gold, "pred": pred, "raw": raw.strip()[:200],
                   "correct": pred == gold, "sec": round(time.time() - t0, 2)}
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"[{n + 1}/{len(rows)}] {r['level2']:<32} gold={gold} pred={pred} {'OK' if rec['correct'] else 'X'}",
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--tag", default="qwen3vl8b")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
