"""Stage 3: fit the belief memory on cached evidence, compare with baselines, write a report.

Two splits, because the sample has only 7 episodes:
  sim->real  fit on the 2 simulated episodes (THOR, VirtualHome; 21 q), test on 5 real (50 q).
             This is the proposal's protocol, at toy scale.
  LOEO       leave one episode out, 7 folds; predictions pooled. Reported on real questions.
Losses (method_v2 §6): value cross-entropy on Known questions only, determinability BCE
(class-balanced), L2 pull toward the initial parameters. No matched-pair loss: the sample has no
matched episodes.

Usage: .venv/bin/python methods/belief/fit_eval.py --tag qwen3vl8b
"""
import argparse, json, math, os, random, sys
from collections import defaultdict
from dataclasses import replace

import torch

sys.path.insert(0, os.path.dirname(__file__))
from memory import BeliefMemory, Config, decide, pack, run_batch  # noqa: E402

HERE = os.path.dirname(__file__)
SIM = {"storm_sim", "storm_vhome"}
LET = "ABCD"


def load(tag):
    return [json.loads(l) for l in open(os.path.join(HERE, "cache", f"{tag}.jsonl"))]


def episode(r):
    return r["id"].rsplit("_q", 1)[0]


# ---------------------------------------------------------------- fitting
def fit(train, cfg, steps=300, lam_reg=0.05, seed=0):
    torch.manual_seed(seed)
    m = BeliefMemory()
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    P = pack(train)
    known = torch.tensor([r["status"] == "known" for r in train])
    gold = torch.tensor([r["values"].index(r["answer_index"]) if r["answer_index"] in r["values"] else -1
                         for r in train])
    use_v = known & (gold >= 0)
    n_kn, n_un = int(known.sum()), int((~known).sum())
    w = torch.where(known, torch.tensor(len(train) / (2 * max(n_kn, 1))), torch.tensor(len(train) / (2 * max(n_un, 1))))
    for _ in range(steps):
        opt.zero_grad()
        b, d = run_batch(m, P, cfg)
        loss = -(torch.log(b[use_v, gold[use_v]] + 1e-6)).sum()
        if cfg.use_dhead:
            d = d.clamp(1e-5, 1 - 1e-5)
            loss = loss - (w * torch.where(known, torch.log(d), torch.log(1 - d))).sum()
        loss = loss / len(train) + lam_reg * m.prior_penalty()
        loss.backward()
        opt.step()
    return m


def selfcheck(rows, cfg):
    """run_batch must reproduce the per-question streaming run() (without inspection)."""
    m = BeliefMemory()
    with torch.no_grad():
        m.log_lam.fill_(math.log(0.05)); m.d_w.copy_(torch.linspace(-1, 1, 8))
        b, d = run_batch(m, pack(rows), cfg)
        for i, r in enumerate(rows):
            o = m.run(r, replace(cfg, use_inspect=False))
            k = len(r["values"])
            assert torch.allclose(o["b"], b[i, :k], atol=1e-5), (r["id"], o["b"], b[i])
            assert abs(o["d"].item() - d[i].item()) < 1e-5, (r["id"], o["d"], d[i])


# ---------------------------------------------------------------- systems
def softmax(d):
    mx = max(d.values())
    z = {k: math.exp(v - mx) for k, v in d.items()}
    s = sum(z.values())
    return {k: v / s for k, v in z.items()}


def p1_pred(r):
    p = softmax(r["p1"]["logp"])
    L = max(p, key=p.get)
    opt = r["p1"]["perm"][LET.index(L)]
    return opt, p[L]


def sys_p1(r):
    opt, _ = p1_pred(r)
    return opt, ("uncertain" if opt in r["abstain"] else "known"), 1


def sys_p1_thresh(tau):
    def f(r):
        opt, pmax = p1_pred(r)
        if opt in r["abstain"] or pmax < tau:
            return (r["abstain"][0] if r["abstain"] else opt), "uncertain", 1
        return opt, "known", 1
    return f


def sys_latest(r):
    lp = dict(list(r["clips"][-1]["logp"].items())[:-1])
    p = softmax(lp)
    return r["values"]["ABCDE".index(max(p, key=p.get))], "known", len(r["clips"])


def sys_belief(m, cfg):
    def f(r):
        with torch.no_grad():
            out = m.run(r, cfg)
        ans, st = decide(r, out, cfg)
        f.extra[r["id"]] = {"d": out["d"].item(), "inspected": out["inspected"], "trace": out["trace"]}
        return ans, st, out["calls"]
    f.extra = {}
    return f


# ---------------------------------------------------------------- metrics
def bucket(c):
    return "low" if c <= 3 else "mid" if c <= 6 else "high"


def br(rows, joint, cell):
    g = defaultdict(list)
    for r in rows:
        g[cell(r)].append(joint[r["id"]])
    vals = [(sum(v) + 1) / (len(v) + 2) for v in g.values()]
    return len(vals) / sum(1 / v for v in vals)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def metrics(rows, preds, ci=True):
    kn = [r for r in rows if r["status"] == "known"]
    un = [r for r in rows if r["status"] == "uncertain"]
    joint = {r["id"]: float(preds[r["id"]][0] == r["answer_index"] and preds[r["id"]][1] == r["status"])
             for r in rows}
    rnd = random.Random(0)
    bs = sorted(br([rnd.choice(rows) for _ in rows], joint, lambda r: r["status"]) for _ in range(2000)) \
        if ci else [float("nan")] * 2000
    return {
        "acc_known": sum(preds[r["id"]][0] == r["answer_index"] for r in kn) / len(kn),
        "false_abst": sum(preds[r["id"]][1] == "uncertain" for r in kn) / len(kn),
        "abst_unc": sum(preds[r["id"]][1] == "uncertain" for r in un) / len(un),
        "joint_unc": sum(joint[r["id"]] for r in un) / len(un),
        "br2": br(rows, joint, lambda r: r["status"]), "br2_ci": (bs[50], bs[1949]),
        "br6": br(rows, joint, lambda r: (r["status"], bucket(r["intensity"]))),
        "calls": sum(preds[r["id"]][2] for r in rows) / len(rows),
    }


def row(name, m, au=None):
    a = f"{au:.2f}" if au is not None and not math.isnan(au) else "–"
    return (f"| {name} | {m['acc_known']:.2f} | {m['false_abst']:.2f} | {m['abst_unc']:.2f} | "
            f"{m['br2']:.2f} [{m['br2_ci'][0]:.2f}, {m['br2_ci'][1]:.2f}] | {m['br6']:.2f} | {a} | {m['calls']:.1f} |")


HEAD = ("| system | acc Known | false abstention (Known) | abstains on Uncertain | BR 2-cell [95% CI] | BR 6-cell | AUROC (1−d) | VLM calls / q |\n"
        "|---|---|---|---|---|---|---|---|")


def tune_tau(train):
    best = (-1, 0.0)
    for tau in [i / 20 for i in range(21)]:
        f = sys_p1_thresh(tau)
        preds = {r["id"]: f(r) for r in train}
        best = max(best, (metrics(train, preds, ci=False)["br2"], tau))
    return best[1]


VARIANTS = {
    "Belief memory (full)": Config(),
    "– no inspection": Config(use_inspect=False),
    "– no drift (λ = 0)": Config(use_drift=False),
    "– no reliability weight (r = 1)": Config(use_reliability=False),
    "– d = max belief (no d-head)": Config(use_dhead=False),
}


def evaluate(train_sets, test_sets, all_rows, title):
    """train_sets/test_sets: lists of row lists (one pair per fold). Predictions pooled over folds."""
    lines = [f"### {title}", "", HEAD]
    pooled = defaultdict(dict)
    extras = {}
    aus = defaultdict(lambda: ([], []))
    params = []
    for train, test in zip(train_sets, test_sets):
        for r in test:
            pooled["Baseline: latest observation"][r["id"]] = sys_latest(r)
            pooled["Baseline: P1 full context (STORM protocol)"][r["id"]] = sys_p1(r)
        tau = tune_tau(train)
        f = sys_p1_thresh(tau)
        for r in test:
            pooled[f"Baseline: P1 + confidence threshold (τ fit on train)"][r["id"]] = f(r)
        untrained = BeliefMemory()
        g = sys_belief(untrained, Config())
        for r in test:
            pooled["Belief memory, initial parameters (no fit)"][r["id"]] = g(r)
        for name, cfg in VARIANTS.items():
            m = fit(train, replace(cfg, use_inspect=False))
            if name == "Belief memory (full)":
                params.append({"lam": math.exp(m.log_lam.item()), "T": math.exp(m.log_T.item()),
                               "rel_a": m.rel_a.item(), "rel_c": m.rel_c.item(),
                               "d_w": [round(x, 2) for x in m.d_w.tolist()], "d_b": round(m.d_b.item(), 2)})
            g = sys_belief(m, cfg)
            for r in test:
                pooled[name][r["id"]] = g(r)
                aus[name][0 if r["status"] == "uncertain" else 1].append(1 - g.extra[r["id"]]["d"])
            if name == "Belief memory (full)":
                extras.update(g.extra)
    for name, preds in pooled.items():
        lines.append(row(name, metrics(all_rows, preds), auroc(*aus[name]) if name in aus else None))
    return "\n".join(lines), pooled, extras, params


def main(args):
    rows = load(args.tag)
    for cfg in VARIANTS.values():
        selfcheck(rows, cfg)
    real = [r for r in rows if r["subset"] not in SIM]
    sim = [r for r in rows if r["subset"] in SIM]
    out = [f"# Belief memory on STORM-Bench sample episodes ({args.tag})", "",
           f"Cached questions: {len(rows)} ({len(real)} real: {sum(r['status']=='known' for r in real)} Known, "
           f"{sum(r['status']=='uncertain' for r in real)} Uncertain; {len(sim)} simulated). "
           "All numbers below are on the real-video questions.", ""]
    report_rows = {}
    if sim and real:
        txt, pooled, extras, params = evaluate([sim], [real], real, "Split 1: fit on simulation, test on real video")
        out += [txt, "", f"Fitted parameters: `{json.dumps(params[0])}`", ""]
        report_rows["sim2real"] = (pooled, extras)
    eps = sorted({episode(r) for r in rows})
    if len(eps) > 1:
        tr = [[r for r in rows if episode(r) != e] for e in eps]
        te = [[r for r in rows if episode(r) == e and r["subset"] not in SIM] for e in eps]
        txt, pooled, extras, _ = evaluate(tr, te, real, "Split 2: leave one episode out (7 folds, pooled)")
        out += [txt, ""]
        report_rows["loeo"] = (pooled, extras)
    # per-question dump for inspection
    dump = []
    for split, (pooled, extras) in report_rows.items():
        for r in real:
            dump.append({"split": split, "id": r["id"], "subset": r["subset"], "type": r["type"],
                         "status": r["status"], "gold": r["answer_index"],
                         **{k: list(v[r["id"]][:2]) for k, v in pooled.items() if r["id"] in v},
                         **extras.get(r["id"], {})})
    with open(os.path.join(HERE, "results", f"{args.tag}_predictions.jsonl"), "w") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    text = "\n".join(out)
    with open(os.path.join(HERE, "results", f"{args.tag}_tables.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3vl8b")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    main(ap.parse_args())
