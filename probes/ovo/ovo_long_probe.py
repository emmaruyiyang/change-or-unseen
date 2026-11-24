"""Perception go/no-go probes (P0-P3, as probes/storm/probe.py) on OVO-Bench backward questions in long videos.

Questions: data/ovo/long_backward.jsonl (EPM / ASI / HLD asked at t >= 120 s; frames and CLIP from
methods/stamem/longdata.py). OVO has no evidence annotation for these tasks, so P2's evidence is retrieved:

  P0  blind (options only)
  P1  online: 32 frames spread uniformly over [0, t] (newest included)
  P2  retrieved evidence: the K frames before t whose CLIP image embedding best matches the text
      "question + gold answer" (at least GAP s apart). Uses the gold answer, so it is an approximate oracle.
  P3  describe-then-answer on the P2 frames (same prompts as probes/storm/probe.py)
  P3o describe-then-answer on the P1 frames (no oracle); run for HLD, whose gold is always "Unable to answer"
      and which has no evidence to retrieve (P2 / P3 are skipped there)

--data meminstream runs the same probes on the MemInStream sample (probes/meminstream/fetch.py), whose questions come
with annotated evidence spans: there P2 / P3 use the annotated frames (at most 8), not retrieval.

Usage: .venv/bin/python probes/ovo/ovo_long_probe.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit --tag ovolong_qwen3vl8b
       .venv/bin/python probes/ovo/ovo_long_probe.py --evidence-only   # write the retrieved frames for spot checks
"""
import argparse, json, os, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # repo root
sys.path.insert(0, os.path.join(ROOT, "probes", "storm"))
sys.path.insert(0, os.path.join(ROOT, "methods", "stamem"))
from probe import MCQ_TAIL, VLM, permutation  # noqa: E402
import longdata  # noqa: E402
from answer import spread  # noqa: E402

K, GAP = 4, 3.0
MAX_EV = 8  # annotated evidence (MemInStream): at most this many frames, spread over the spans
LETTERS = "ABCDE"  # MemInStream has five options


def parse_letter(text):
    m = re.search(r"\b([A-E])\b", text.strip().upper())
    return m.group(1) if m else None


def annotated(times, q):
    """Frames inside the annotated evidence spans (gonogo's evidence_times), thinned to MAX_EV."""
    grid = [t for t in times if t <= q["query_time"]]
    pick = set()
    for a, b in q["evidence_spans"]:
        inside = [t for t in grid if a - 0.5 <= t <= b + 0.5]
        pick.update(inside or [min(grid, key=lambda t: abs(t - (a + b) / 2))])
    return spread(sorted(pick), q["query_time"], MAX_EV)


def text_embed(texts):
    import open_clip, torch
    from stream import _model
    m, _, _ = _model()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        T = m.encode_text(tok(texts)).float()
        return torch.nn.functional.normalize(T, dim=-1).numpy()


def retrieve(E, times, upto, t_emb):
    """Top-K frames before `upto` by CLIP similarity to the text, kept at least GAP s apart."""
    idx = [i for i, t in enumerate(times) if t <= upto]
    sims = E[idx] @ t_emb
    pick = []
    for j in np.argsort(-sims):
        t = times[idx[j]]
        if all(abs(t - p) >= GAP for p, _ in pick):
            pick.append((t, float(sims[j])))
        if len(pick) == K:
            break
    return sorted(pick)


def evidence(eps, data):
    EVID = os.path.join(HERE, "results", f"{data}_evidence.json".replace("ovo_backward", "ovolong"))
    if data == "meminstream":
        return {q["id"]: [(t, None) for t in annotated(e["times"], q)] for e in eps.values() for q in e["questions"]}
    if os.path.exists(EVID):
        return json.load(open(EVID))
    out = {}
    for e in eps.values():
        E = longdata.embed(e)
        qs = [q for q in e["questions"] if q["question_type"] != "HLD"]
        if not qs:
            continue
        T = text_embed([f"{q['question']} {q['options'][q['answer_index']]}" for q in qs])
        for q, t_emb in zip(qs, T):
            out[q["id"]] = retrieve(E, e["times"], q["query_time"], t_emb)
    os.makedirs(os.path.dirname(EVID), exist_ok=True)
    json.dump(out, open(EVID, "w"), indent=0)
    return out


def run(args):
    eps = longdata.episodes(args.data)
    ev = evidence(eps, args.data)
    if args.evidence_only:
        print(f"{len(ev)} questions with evidence")
        return
    tag = args.tag or ("ovolong_qwen3vl8b" if args.data == "ovo_backward" else f"{args.data}_qwen3vl8b")
    out_path = os.path.join(HERE, "results", f"{tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"], r["probe"]) for r in map(json.loads, open(out_path))}
    vlm = VLM(args.model)
    qs = [(e, q) for e in eps.values() for q in e["questions"]
          if not args.types or q["question_type"] in args.types][:args.limit]
    with open(out_path, "a") as fo:
        for n, (e, q) in enumerate(qs):
            k = len(q["options"])
            perm = permutation(q["id"], k)
            gold = LETTERS[perm.index(q["answer_index"])]
            opts = "\n".join(f"{LETTERS[j]}. {q['options'][i]}" for j, i in enumerate(perm))
            now = f"The current time is t={q['query_time']:g}s."
            stem = f"Question: {q['question']}\nOptions:\n{opts}\n{MCQ_TAIL}"
            on_t = spread(e["times"], q["query_time"])
            ev_t = [t for t, _ in ev.get(q["id"], [])]
            hld = q["question_type"] == "HLD"
            probes = ["P0", "P1", "P3o"] if hld else ["P0", "P1", "P2", "P3"]
            for probe in probes:
                if (q["id"], probe) in done or probe not in args.probes:
                    continue
                t0 = time.time()
                if probe in ("P3", "P3o"):
                    ts = ev_t if probe == "P3" else on_t
                    intro = ("These frames are selected moments from a video stream." if probe == "P3" else
                             "These are frames of a video stream seen so far, spread evenly over time.")
                    note = vlm.ask([e["frame"](t) for t in ts], ts,
                                   f"{intro} {now}\nSomeone will later ask: \"{q['question']}\"\n"
                                   "Write down, in at most three sentences, the visual facts in these frames "
                                   "needed to answer it: objects, locations, states, counts and when they were seen. "
                                   "If the frames do not show enough to decide, say so.", 160)
                    raw = vlm.ask([], [], f"Notes taken while watching a video stream ({now}):\n{note.strip()}\n"
                                          f"Using only these notes, answer.\n{stem}", 8)
                    extra = {"note": note.strip(), "n_frames": len(ts)}
                else:
                    ts, prompt = {
                        "P0": ([], f"You will be asked about a video you cannot see. Make your best guess.\n{stem}"),
                        "P1": (on_t, "These are frames of a video stream seen so far, spread evenly over time. "
                                     f"{now}\n{stem}"),
                        "P2": (ev_t, f"These frames are selected moments from a video stream. {now}\n{stem}"),
                    }[probe]
                    raw = vlm.ask([e["frame"](t) for t in ts], ts, prompt, 8)
                    extra = {"n_frames": len(ts)}
                pred = parse_letter(raw)
                abst = [LETTERS[perm.index(i)] for i, o in enumerate(q["options"])
                        if re.search(r"unable to answer|have not seen|did not see|never", o, re.I)]
                rec = {"id": q["id"], "video": e["video"].split("videos/")[-1], "probe": probe,
                       "type": q["question_type"], "level1": q.get("level1"), "query_time": q["query_time"], "gold": gold, "pred": pred,
                       "raw": raw.strip()[:200], "abstain_letters": abst, "correct": pred == gold,
                       "sec": round(time.time() - t0, 2), **extra}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                print(f"[{n + 1}/{len(qs)}] {probe:<3} {q['question_type']} gold={gold} pred={pred} "
                      f"{'OK ' if rec['correct'] else 'X  '} {rec['sec']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--data", default="ovo_backward", choices=["ovo_backward", "meminstream"])
    ap.add_argument("--tag", default=None, help="default: ovolong_qwen3vl8b / <data>_qwen3vl8b")
    ap.add_argument("--probes", nargs="+", default=["P0", "P1", "P2", "P3", "P3o"])
    ap.add_argument("--types", nargs="+", help="only these question types (EPM / ASI / HLD)")
    ap.add_argument("--limit", type=int, default=None, help="first N questions after the type filter")
    ap.add_argument("--evidence-only", action="store_true")
    run(ap.parse_args())
