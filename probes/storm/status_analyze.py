"""Score the answerability probes: abstention on both sides, AUROC, joint correctness, BR."""
import json, os, random, sys
from collections import defaultdict

RES = os.path.join(os.path.dirname(__file__), "results")
B = 5000


def bucket(c):
    return "low" if c <= 3 else "mid" if c <= 6 else "high"


def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def br(items, joint, cells):
    """STORM-BR: harmonic mean of Laplace-smoothed joint accuracy over occupied cells."""
    groups = defaultdict(list)
    for r in items:
        groups[cells(r)].append(joint(r))
    vals = [(sum(v) + 1) / (len(v) + 2) for v in groups.values()]
    return len(vals) / sum(1 / v for v in vals)


def boot(items, fn, seed=0):
    rnd = random.Random(seed)
    v = sorted(fn([rnd.choice(items) for _ in items]) for _ in range(B))
    return v[int(.025 * B)], v[int(.975 * B)]


def report(tag):
    rows = [json.loads(l) for l in open(os.path.join(RES, f"status_{tag}.jsonl"))]
    p1 = {}
    pf = os.path.join(RES, f"{tag}.jsonl")
    if os.path.exists(pf):
        p1 = {r["id"]: r["pred"] for r in map(json.loads, open(pf)) if r["probe"] == "P1"}
    kn = [r for r in rows if r["status"] == "known"]
    un = [r for r in rows if r["status"] == "uncertain"]
    st = lambda r, k: "uncertain" if r[f"{k}_p_unc"] > 0.5 else "known"
    out = [f"## {tag}  (n = {len(rows)}: {len(kn)} Known, {len(un)} Uncertain)", "",
           "| status probe | overconfidence (Uncertain → Known) | false abstention (Known → Uncertain) | balanced status acc | AUROC of P(uncertain) [95% CI] |",
           "|---|---|---|---|---|"]
    for k, name in [("s1", "S1 STORM probe (frames)"), ("s2", "S2 describe → judge (notes)")]:
        oc = sum(st(r, k) == "known" for r in un) / len(un)
        fa = sum(st(r, k) == "uncertain" for r in kn) / len(kn)
        a = auroc([r[f"{k}_p_unc"] for r in un], [r[f"{k}_p_unc"] for r in kn])
        lo, hi = boot(rows, lambda s: auroc([r[f"{k}_p_unc"] for r in s if r["status"] == "uncertain"],
                                            [r[f"{k}_p_unc"] for r in s if r["status"] == "known"]))
        out.append(f"| {name} | {oc:.2f} ({round(oc*len(un))}/{len(un)}) | {fa:.2f} ({round(fa*len(kn))}/{len(kn)}) "
                   f"| {1 - (oc + fa) / 2:.2f} | {a:.2f} [{lo:.2f}, {hi:.2f}] |")
    out.append("")

    def joint(answer_of, status_key, gate):
        def j(r):
            s = st(r, status_key)
            ans = answer_of(r)
            if gate and s == "uncertain" and r["abstain_letters"]:
                ans = r["abstain_letters"][0]
            return float(s == r["status"] and ans == r["gold"])
        return j

    systems = []
    if p1:
        systems += [("P1 answer + S1 status (STORM protocol)", joint(lambda r: p1.get(r["id"]), "s1", False)),
                    ("P1 answer, abstain when S1 says Uncertain", joint(lambda r: p1.get(r["id"]), "s1", True))]
    systems += [("S2 pipeline: note → answer + note → status", joint(lambda r: r["s2_answer"], "s2", False)),
                ("S2 pipeline, abstain when S2 says Uncertain", joint(lambda r: r["s2_answer"], "s2", True))]
    if p1:
        systems += [("P1 answer, abstain when S2 says Uncertain", joint(lambda r: p1.get(r["id"]), "s2", True))]
    out += ["| system | joint acc Known | joint acc Uncertain | BR, 2 cells [95% CI] | BR, 6 cells (STORM grid) |",
            "|---|---|---|---|---|"]
    two = lambda r: r["status"]
    six = lambda r: (r["status"], bucket(r["intensity"]))
    for name, j in systems:
        jk = sum(map(j, kn)) / len(kn)
        ju = sum(map(j, un)) / len(un)
        lo, hi = boot(rows, lambda s: br(s, j, two))
        out.append(f"| {name} | {jk:.2f} | {ju:.2f} | {br(rows, j, two):.2f} [{lo:.2f}, {hi:.2f}] | {br(rows, j, six):.2f} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["qwen25vl3b", "qwen3vl8b"]:
        if os.path.exists(os.path.join(RES, f"status_{t}.jsonl")):
            print(report(t))
