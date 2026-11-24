"""Baseline A (and blind P0) on a small OVO-Bench sample, same protocol as probe.py.

Only the multiple-choice tasks (backward: EPM ASI HLD; real-time: STU OJR ATR ACR OCR FPD).
Forward-active tasks (REC SSR CRR) are multi-trigger and are not scored here.

Usage:
  python probes/ovo/ovo_probe.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit --tag ovo_qwen3vl8b
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storm"))
import probe
from probe import LETTERS, MAX_FRAMES, MCQ_TAIL, VLM, duration, frame_path, parse_letter, permutation

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ovo")
probe.FRAMES = os.path.join(ROOT, "frames")
MCQ_TASKS = {"EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"}


def online_times(query_time, dur):
    """1 FPS frames with timestamp <= query_time; above MAX_FRAMES, uniform subsample keeping the newest."""
    last = min(query_time, dur - 0.05)
    ts = [float(k) for k in range(int(last) + 1)]
    if len(ts) > MAX_FRAMES:
        idx = [round(i * (len(ts) - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        ts = [ts[i] for i in sorted(set(idx))]
    return ts


def run(args):
    qs = [json.loads(l) for l in open(os.path.join(ROOT, "samples.jsonl"))]
    qs = [q for q in qs if q["task"] in MCQ_TASKS]
    out_path = os.path.join(os.path.dirname(__file__), "results", f"{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["probe"]) for r in map(json.loads, open(out_path))}
    vlm = VLM(args.model)
    durs = {}
    with open(out_path, "a") as fo:
        for n, q in enumerate(qs):
            video = os.path.join(ROOT, "videos", q["local"])
            dur = durs.setdefault(video, duration(video))
            k = len(q["options"])
            perm = permutation(str(q["id"]), k)
            gold = LETTERS[perm.index(q["gt"])]
            opts = "\n".join(f"{LETTERS[j]}. {q['options'][i]}" for j, i in enumerate(perm))
            stem = f"Question: {q['question']}\nOptions:\n{opts}\n{MCQ_TAIL}"
            now = f"The current time is t={q['realtime']:g}s."
            times = online_times(q["realtime"], dur)
            jobs = {
                "P0": ([], f"You will be asked about a video you cannot see. Make your best guess.\n{stem}"),
                "P1": (times, f"These are frames of a video stream seen so far, sampled at one per second "
                              f"(uniformly thinned to at most {MAX_FRAMES}). {now}\n{stem}"),
            }
            for p in args.probes:
                if (q["id"], p) in done:
                    continue
                t0 = time.time()
                ts, prompt = jobs[p]
                raw = vlm.ask([frame_path(video, t) for t in ts], ts, prompt, 8)
                pred = parse_letter(raw)
                rec = {"id": q["id"], "task": q["task"], "probe": p, "gold": gold, "pred": pred,
                       "correct": pred == gold, "raw": raw.strip()[:200], "n_frames": len(ts),
                       "realtime": q["realtime"], "video_dur": round(dur, 1), "sec": round(time.time() - t0, 2)}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(f"[{n+1}/{len(qs)}] {p} {q['task']} gold={gold} pred={pred} "
                      f"{'OK ' if rec['correct'] else 'X  '} frames={len(ts)} {rec['sec']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probes", nargs="+", default=["P0", "P1"])
    run(ap.parse_args())
