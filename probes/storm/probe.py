"""Perception go/no-go probes (P0-P3) on STORM-Bench example episodes.

Usage:
  python probes/storm/probe.py --model mlx-community/Qwen2.5-VL-3B-Instruct-bf16 --tag qwen25vl3b
See PREREGISTRATION.md for what each probe measures.
"""
import argparse, glob, hashlib, json, math, os, re, subprocess, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
DATA = os.path.join(ROOT, "data", "storm", "examples")
FRAMES = os.path.join(ROOT, "data", "storm", "frames")
LETTERS = "ABCD"
MAX_FRAMES = 32
MAX_SIDE = 448


def load_questions():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", "*.questions.jsonl"))):
        subset = os.path.basename(os.path.dirname(f))
        video = f.replace(".questions.jsonl", ".mp4")
        for line in open(f):
            r = json.loads(line)
            r["subset"], r["video"] = subset, video
            rows.append(r)
    return rows


def duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video], capture_output=True, text=True).stdout
    return float(out.strip())


def frame_path(video, t):
    d = os.path.join(FRAMES, os.path.splitext(os.path.basename(video))[0])
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"t{t:05.1f}.jpg")
    if not os.path.exists(p):
        vf = f"scale='if(gt(iw,ih),{MAX_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_SIDE})'"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                        "-vf", vf, "-q:v", "3", "-y", p], check=True)
    return p


def online_times(q, dur):
    """STORM online protocol: 1 FPS frames with timestamp <= query_time."""
    last = min(q["query_time"], dur - 0.05)
    ts = [float(k) for k in range(int(math.floor(last)) + 1)]
    if len(ts) > MAX_FRAMES:  # uniform subsample, always keep the newest frame
        idx = [round(i * (len(ts) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        ts = [ts[i] for i in sorted(set(idx))]
    return ts


def evidence_times(q, dur):
    grid = online_times({"query_time": q["query_time"]}, dur) if False else \
        [float(k) for k in range(int(math.floor(min(q["query_time"], dur - 0.05))) + 1)]
    pick = set()
    for a, b in q["evidence_spans"]:
        inside = [t for t in grid if a - 0.5 <= t <= b + 0.5]
        pick.update(inside or [min(grid, key=lambda t: abs(t - (a + b) / 2))])
    ts = sorted(pick)
    if len(ts) > MAX_FRAMES:
        idx = [round(i * (len(ts) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        ts = [ts[i] for i in sorted(set(idx))]
    return ts


def permutation(qid, n=4):
    """Fixed per-question option order, derived from the id."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{qid}:{i}".encode()).hexdigest())


def options_block(q, perm):
    return "\n".join(f"{LETTERS[j]}. {q['options'][i]}" for j, i in enumerate(perm))


def parse_letter(text):
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


class VLM:
    def __init__(self, model_id):
        from mlx_vlm import load
        self.model, self.processor = load(model_id)

    def ask(self, frames, times, text, max_tokens):
        from mlx_vlm import generate
        content = []
        for t in times:
            content.append({"type": "text", "text": f"Frame at t={t:.0f}s:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        tok = getattr(self.processor, "tokenizer", self.processor)
        try:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        res = generate(self.model, self.processor, prompt, image=frames or None,
                       max_tokens=max_tokens, temperature=0.0, verbose=False)
        import mlx.core as mx
        mx.clear_cache()  # without this the Metal cache grows across calls and the run ends up in swap
        return res.text if hasattr(res, "text") else str(res)


MCQ_TAIL = "Answer with the letter of the correct option only."


def run(args):
    qs = load_questions()
    if args.subsets:
        qs = [q for q in qs if q["subset"] in args.subsets]
    if args.limit:
        qs = qs[: args.limit]
    out_path = os.path.join(os.path.dirname(__file__), "results", f"{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["probe"]) for r in map(json.loads, open(out_path))}
    vlm = VLM(args.model)
    durs = {}
    with open(out_path, "a") as fo:
        for n, q in enumerate(qs):
            dur = durs.setdefault(q["video"], duration(q["video"]))
            perm = permutation(q["id"])
            gold = LETTERS[perm.index(q["answer_index"])]
            opts = options_block(q, perm)
            now = f"The current time is t={q['query_time']:g}s."
            stem = f"Question: {q['question']}\nOptions:\n{opts}\n{MCQ_TAIL}"
            on_t, ev_t = online_times(q, dur), evidence_times(q, dur)
            jobs = {
                "P0": ([], f"You will be asked about a video you cannot see. Make your best guess.\n{stem}"),
                "P1": (on_t, f"These are the frames of a video stream seen so far, one per second. {now}\n{stem}"),
                "P2": (ev_t, f"These frames are selected moments from a video stream. {now}\n{stem}"),
            }
            for probe in args.probes:
                if (q["id"], probe) in done:
                    continue
                t0 = time.time()
                if probe == "P3":
                    frames = [frame_path(q["video"], t) for t in ev_t]
                    note = vlm.ask(frames, ev_t,
                                   f"These frames are selected moments from a video stream. {now}\n"
                                   f"Someone will later ask: \"{q['question']}\"\n"
                                   "Write down, in at most three sentences, the visual facts in these frames "
                                   "needed to answer it: objects, locations, states, counts and when they were seen. "
                                   "If the frames do not show enough to decide, say so.", 160)
                    raw = vlm.ask([], [], f"Notes taken while watching a video stream ({now}):\n{note.strip()}\n"
                                          f"Using only these notes, answer.\n{stem}", 8)
                    extra = {"note": note.strip(), "n_frames": len(ev_t)}
                else:
                    times, prompt = jobs[probe]
                    frames = [frame_path(q["video"], t) for t in times]
                    raw = vlm.ask(frames, times, prompt, 8)
                    extra = {"n_frames": len(times)}
                pred = parse_letter(raw)
                abst = [LETTERS[perm.index(i)] for i, o in enumerate(q["options"])
                        if re.search(r"cannot be determined|uncertain|unresolved|no definite|not uniquely|"
                                     r"no concrete alternative", o, re.I)]
                rec = {"id": q["id"], "subset": q["subset"], "probe": probe, "type": q["question_type"],
                       "status": q["diagnostics"]["epistemic_status"], "gold": gold, "pred": pred,
                       "raw": raw.strip()[:200], "abstain_letters": abst, "correct": pred == gold,
                       "sec": round(time.time() - t0, 2), **extra}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(f"[{n+1}/{len(qs)}] {probe} {q['subset']:<14} gold={gold} pred={pred} "
                      f"{'OK ' if rec['correct'] else 'X  '} {rec['sec']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", nargs="+", default=["P0", "P1", "P2", "P3"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subsets", nargs="+", default=None)
    run(ap.parse_args())
