"""Score the OVO-Bench Baseline A sample: accuracy by probe, by task group and by task, with latency."""
import json, os, random, statistics as st, sys

GROUP = {"EPM": "backward", "ASI": "backward", "HLD": "backward", "STU": "realtime", "OJR": "realtime",
         "ATR": "realtime", "ACR": "realtime", "OCR": "realtime", "FPD": "realtime"}
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results", "ovo_qwen3vl8b.jsonl")
rows = [json.loads(l) for l in open(path)]


def acc(rs):
    return sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")


def ci(rs, n=2000):
    random.seed(0)
    xs = sorted(acc(random.choices(rs, k=len(rs))) for _ in range(n))
    return xs[int(.025 * n)], xs[int(.975 * n)]


for p in ["P0", "P1"]:
    rs = [r for r in rows if r["probe"] == p]
    lo, hi = ci(rs)
    print(f"{p} all n={len(rs)} acc={acc(rs):.2f} [{lo:.2f}, {hi:.2f}]")
    for g in ["backward", "realtime"]:
        sub = [r for r in rs if GROUP[r["task"]] == g]
        print(f"   {g:<9} n={len(sub):2d} acc={acc(sub):.2f}")
    for t in GROUP:
        sub = [r for r in rs if r["task"] == t]
        if sub:
            print(f"      {t} n={len(sub)} acc={acc(sub):.2f}")
p1 = [r for r in rows if r["probe"] == "P1"]
secs = sorted(r["sec"] for r in p1)
print(f"P1 latency median {st.median(secs):.1f}s, p90 {secs[int(.9 * len(secs)) - 1]:.1f}s, "
      f"mean frames {st.mean(r['n_frames'] for r in p1):.1f}, "
      f"mean query time {st.mean(r['realtime'] for r in p1):.0f}s")
