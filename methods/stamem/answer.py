"""Answer STORM questions from what each policy kept under a frame budget K (the accuracy metric).

The VLM sees only the kept frames (plus, for StaMem, a one-line tag per frame) and the question,
asked at the query time. Systems:
  full          all 1 fps frames up to the query time (capped at 32, STORM protocol; upper reference)
  fifo / uniform / dedup / stamem         K kept frames, timestamps only
  stamem_tags   StaMem frames plus their state tags (since / confirmed / replaced)

--data ovo | videomme runs the long-video headroom check (longdata.py); there "full" is 32 frames spread
uniformly over the prefix, the usual long-video protocol.

Usage: .venv/bin/python methods/stamem/answer.py --K 4 --model mlx-community/Qwen2.5-VL-3B-Instruct-bf16 --tag q25vl3b
       .venv/bin/python methods/stamem/answer.py --data ovo --K 8
Then:  .venv/bin/python methods/stamem/answer.py --summarize --tag q25vl3b [--data ovo]
"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import make  # noqa: E402
import longdata  # noqa: E402
import stream  # noqa: E402
from probe import LETTERS, MAX_FRAMES, MCQ_TAIL, VLM, frame_path, online_times, options_block, parse_letter, permutation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = {"storm_sim", "storm_vhome"}
SYSTEMS = ["full", "fifo", "uniform", "dedup", "stamem", "stamem_tags"]


def tag(m):
    s = f"Frame at t={m['t']:.0f}s"
    if m["confirmed"] > m["t"]:
        s += f" (this view still looked the same at t={m['confirmed']:.0f}s)"
    if not m["current"]:
        s += f" (earlier state: this view had changed by t={m['replaced_at']:.0f}s)"
    return s + ":"


def spread(times, upto, n=MAX_FRAMES):
    """n frames spread uniformly over [0, upto], newest included (the long-video "full" reference)."""
    ts = [t for t in times if t <= upto]
    if len(ts) <= n:
        return ts
    return [ts[round(i * (len(ts) - 1) / (n - 1))] for i in range(n)]


def out_path(args):
    pre = "" if args.data == "storm" else f"{args.data}_"
    return os.path.join(HERE, "results", f"answers_{pre}{args.tag}_K{args.K}.jsonl")


def ask(vlm, e, entries, labels, text):
    frames = [e["frame"](m["t"]) if "frame" in e else frame_path(e["video"], m["t"]) for m in entries]
    content = []
    for lab in labels:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    from mlx_vlm import generate
    import mlx.core as mx
    msgs = [{"role": "user", "content": content}]
    prompt = vlm.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    res = generate(vlm.model, vlm.processor, prompt, image=frames, max_tokens=8, temperature=0.0, verbose=False)
    mx.clear_cache()
    return res.text if hasattr(res, "text") else str(res)


def run(args):
    out = out_path(args)
    done = set()
    if os.path.exists(out):
        done = {(r["id"], r["system"]) for r in map(json.loads, open(out))}
    vlm = VLM(args.model)
    long = args.data != "storm"
    eps = longdata.episodes(args.data) if long else stream.episodes()
    with open(out, "a") as fo:
        for k, e in eps.items():
            E = longdata.embed(e) if long else stream.embed(k, e)
            for q in e["questions"]:
                perm = permutation(q["id"], len(q["options"]))
                gold = LETTERS[perm.index(q["answer_index"])]
                now = f"The current time is t={q['query_time']:g}s."
                stem = f"Question: {q['question']}\nOptions:\n{options_block(q, perm)}\n{MCQ_TAIL}"
                n_avail = sum(t <= q["query_time"] for t in e["times"])
                for system in args.systems:
                    if (q["id"], system) in done:
                        continue
                    t0 = time.time()
                    if system == "full":
                        ts = spread(e["times"], q["query_time"]) if long else online_times(q, e["dur"])
                        entries = [{"t": t} for t in ts]
                        labels = [f"Frame at t={m['t']:.0f}s:" for m in entries]
                        intro = ("These are frames of a video stream seen so far, spread evenly over time."
                                 if long else "These are the frames of a video stream seen so far, one per second.")
                    else:
                        p = make(system.replace("_tags", ""), args.K)
                        entries = p.run(e["times"], E, q["query_time"])
                        labels = [tag(m) if system == "stamem_tags" else f"Frame at t={m['t']:.0f}s:"
                                  for m in entries]
                        intro = ("These frames are what a memory kept from a video stream; "
                                 "not every second was kept.")
                    raw = ask(vlm, e, entries, labels, f"{intro} {now}\n{stem}")
                    pred = parse_letter(raw)
                    rec = {"id": q["id"], "subset": e["subset"], "type": q["question_type"],
                           "status": q["status"] if long else q["diagnostics"]["epistemic_status"], "system": system, "K": args.K,
                           "n_avail": n_avail, "kept": [m["t"] for m in entries], "gold": gold, "pred": pred,
                           "correct": pred == gold, "sec": round(time.time() - t0, 2)}
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                    print(f"{q['id'][-28:]:<28} {system:<12} gold={gold} pred={pred} "
                          f"{'OK' if rec['correct'] else 'X '} {rec['sec']}s", flush=True)
                    if args.pause:
                        time.sleep(args.pause)


def summarize(args):
    rows = [json.loads(l) for l in open(out_path(args))]
    if args.data != "storm":
        return summarize_long(rows, args)
    acc = collections.defaultdict(list)
    for r in rows:
        real = "sim" if r["subset"] in SIM else "real"
        acc[(r["system"], real, "all")].append(r["correct"])
        if r["n_avail"] > args.K:
            acc[(r["system"], real, "over")].append(r["correct"])
        if r["status"] == "known" and real == "real":
            acc[(r["system"], "real_known", "all")].append(r["correct"])
    print(f"| system (K={args.K}) | real, all | real, prefix > K | real Known | sim, all |")
    print("|---|---|---|---|---|")
    for s in SYSTEMS:
        def f(key):
            v = acc.get(key, [])
            return f"{sum(v) / len(v):.2f} ({len(v)})" if v else "–"
        print(f"| {s} | {f((s, 'real', 'all'))} | {f((s, 'real', 'over'))} | "
              f"{f((s, 'real_known', 'all'))} | {f((s, 'sim', 'all'))} |")


def summarize_long(rows, args):
    acc = collections.defaultdict(list)
    for r in rows:
        acc[(r["system"], "all")].append(r["correct"])
        acc[(r["system"], r["type"])].append(r["correct"])
    types = sorted({r["type"] for r in rows}, key=lambda t: -len(acc[("full", t)]))[:6]
    print(f"| system ({args.data}, K={args.K}) | all | " + " | ".join(types) + " |")
    print("|---" * (len(types) + 2) + "|")
    for s in SYSTEMS:
        def f(key):
            v = acc.get(key, [])
            return f"{sum(v) / len(v):.2f} ({len(v)})" if v else "–"
        print(f"| {s} | {f((s, 'all'))} | " + " | ".join(f((s, t)) for t in types) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--data", default="storm", choices=["storm", "ovo", "videomme"])
    ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
    ap.add_argument("--tag", default="q25vl3b")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between calls, to keep the Mac cool")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
