"""No-VLM diagnostics for each write/evict policy under a frame budget K (cheap go/no-go).

For every STORM question, run the policy on the 1 fps stream up to the query time and measure
  ev_time   share of the question's evidence spans that a kept frame falls inside (±0.5 s)
  ev_view   share of evidence spans that a kept frame looks like (CLIP cos >= 0.95 to a span frame)
  wasted    share of kept frames that near-duplicate another kept frame (cos >= 0.95)
Only questions whose prefix has more than K frames are counted (otherwise every policy keeps all).

Usage: .venv/bin/python methods/stamem/coverage.py [--K 2 4 8] [--sweep]
"""
import argparse, itertools, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import make  # noqa: E402
from stream import embed, episodes  # noqa: E402

SIM = {"storm_sim", "storm_vhome"}
DUP = 0.95


def score(q, keep, times, E):
    upto = [i for i, t in enumerate(times) if t <= q["query_time"]]
    kt = [m["t"] for m in keep]
    ki = [m["i"] for m in keep]
    spans = q["evidence_spans"] or []
    ev_t, ev_v = [], []
    for a, b in spans:
        inside = [i for i in upto if a - 0.5 <= times[i] <= b + 0.5] or \
                 [min(upto, key=lambda i: abs(times[i] - (a + b) / 2))]
        ev_t.append(any(a - 0.5 <= t <= b + 0.5 for t in kt))
        ev_v.append(any(float(E[i] @ E[j]) >= DUP for i in inside for j in ki))
    sims = E[ki] @ E[ki].T
    np.fill_diagonal(sims, -1)
    wasted = float((sims.max(1) >= DUP).mean()) if len(ki) > 1 else 0.0
    return (np.mean(ev_t) if spans else np.nan, np.mean(ev_v) if spans else np.nan, wasted)


def evaluate(eps, K, policy, subsets=None, **kw):
    rows = []
    for k, e in eps.items():
        if subsets is not None and (e["subset"] in SIM) != (subsets == "sim"):
            continue
        E = embed(k, e)
        for q in e["questions"]:
            n_avail = sum(t <= q["query_time"] for t in e["times"])
            if n_avail <= K:
                continue
            p = make(policy, K, **kw)
            rows.append(score(q, p.run(e["times"], E, q["query_time"]), e["times"], E))
    a = np.array(rows, dtype=float)
    return len(rows), np.nanmean(a[:, 0]), np.nanmean(a[:, 1]), np.nanmean(a[:, 2])


def main(args):
    eps = episodes()
    if args.sweep:
        print("tau sweep for StaMem on the simulated episodes (K=4); pick by ev_view, then check on real")
        for ts, tv in itertools.product([0.90, 0.92, 0.93, 0.95, 0.97], [0.70, 0.75, 0.80, 0.85]):
            if tv >= ts:
                continue
            n, et, ev, w = evaluate(eps, 4, "stamem", "sim", tau_same=ts, tau_view=tv)
            print(f"  tau_same {ts:.2f} tau_view {tv:.2f}: n={n} ev_time {et:.2f} ev_view {ev:.2f} wasted {w:.2f}")
    for split in ["real", "sim"]:
        print(f"\n## {split} episodes")
        print("| K | policy | n | evidence kept (time) | evidence kept (view) | wasted slots |")
        print("|---|---|---|---|---|---|")
        for K in args.K:
            for pol in ["fifo", "uniform", "dedup", "stamem"]:
                n, et, ev, w = evaluate(eps, K, pol, split)
                print(f"| {K} | {pol} | {n} | {et:.2f} | {ev:.2f} | {w:.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--sweep", action="store_true")
    main(ap.parse_args())
