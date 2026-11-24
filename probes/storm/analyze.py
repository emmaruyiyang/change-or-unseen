"""Score probe results against the pre-registered go/no-go rules."""
import json, os, random, sys
from collections import defaultdict

RES = os.path.join(os.path.dirname(__file__), "results")
REAL = {"storm_cook", "storm_bike", "storm_healthy", "storm_music", "storm_sports"}
IN_SCOPE = {"current_state", "object_tracking", "state_change"}
B = 5000


def load(tag):
    by = defaultdict(dict)
    for r in map(json.loads, open(os.path.join(RES, f"{tag}.jsonl"))):
        by[r["id"]][r["probe"]] = r
    return by


def acc(ids, by, p):
    return sum(by[i][p]["correct"] for i in ids) / len(ids)


def ci(ids, fn, seed=0):
    rnd = random.Random(seed)
    vals = sorted(fn([rnd.choice(ids) for _ in ids]) for _ in range(B))
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def fmt(v, lo, hi):
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def report(tag):
    by = load(tag)
    probes = ["P0", "P1", "P2", "P3"]
    ids = [i for i in by if all(p in by[i] for p in probes)]
    first = lambda i: by[i]["P1"]
    real_known = [i for i in ids if first(i)["subset"] in REAL and first(i)["status"] == "known"]
    scope_known = [i for i in real_known if first(i)["type"] in IN_SCOPE]
    real_unc = [i for i in ids if first(i)["subset"] in REAL and first(i)["status"] == "uncertain"]
    sim_known = [i for i in ids if first(i)["subset"] not in REAL and first(i)["status"] == "known"]
    out = [f"## {tag}  (questions scored: {len(ids)})", ""]

    def table(name, sel):
        if not sel:
            return
        out.append(f"**{name}** (n = {len(sel)})")
        out.append("")
        out.append("| probe | accuracy [95% CI] |")
        out.append("|---|---|")
        for p in probes:
            out.append(f"| {p} | {fmt(acc(sel, by, p), *ci(sel, lambda s: acc(s, by, p)))} |")
        out.append("")

    table("Real video, Known (primary)", real_known)
    table("Real video, Known, in-scope types", scope_known)
    table("Simulation, Known", sim_known)

    s = real_known
    a = {p: acc(s, by, p) for p in probes}
    d = lambda x, y: (lambda ss: acc(ss, by, x) - acc(ss, by, y))
    g = {
        "P2": (a["P2"], ci(s, lambda ss: acc(ss, by, "P2")), 0.70),
        "P2-P0": (a["P2"] - a["P0"], ci(s, d("P2", "P0")), 0.20),
        "P2-P1": (a["P2"] - a["P1"], ci(s, d("P2", "P1")), 0.10),
        "P3-P2": (a["P3"] - a["P2"], ci(s, d("P3", "P2")), -0.10),
    }
    out.append("**Decision quantities (real Known)**")
    out.append("")
    out.append("| quantity | value [95% CI] | threshold | point passes | CI clear of threshold |")
    out.append("|---|---|---|---|---|")
    for k, (v, (lo, hi), th) in g.items():
        passes = v >= th
        clear = (lo >= th) or (hi < th)
        out.append(f"| {k} | {fmt(v, lo, hi)} | ≥ {th:+.2f} | {'yes' if passes else 'no'} | {'yes' if clear else 'no'} |")
    out.append("")
    g1_pass = g["P2"][0] >= 0.70 and g["P2-P0"][0] >= 0.20
    g1_clear = all(((g[k][1][0] >= g[k][2]) or (g[k][1][1] < g[k][2])) for k in ("P2", "P2-P0"))
    out.append(f"- G1 perception adequate: {'PASS' if g1_pass else 'FAIL'}"
               f"{'' if g1_clear else ' (inconclusive: CI straddles threshold)'}")
    out.append(f"- G2 memory headroom: {'PASS' if g['P2-P1'][0] >= 0.10 else 'FAIL'}")
    out.append(f"- G3 text memory sufficient: {'PASS' if g['P3-P2'][0] >= -0.10 else 'FAIL'}")
    out.append("")

    if real_unc:
        ab = sum(by[i]["P1"]["pred"] in by[i]["P1"]["abstain_letters"] for i in real_unc)
        out.append(f"- Real Uncertain (n = {len(real_unc)}): P1 picks the abstention option {ab}/{len(real_unc)}")
    kn_with_abst = [i for i in real_known if by[i]["P1"]["abstain_letters"]]
    out.append(f"- Real Known items that even offer an abstention option: {len(kn_with_abst)}")
    per = defaultdict(list)
    for i in real_known:
        per[first(i)["subset"]].append(i)
    out.append("- Per real domain, Known, P1 / P2: " + ", ".join(
        f"{k.replace('storm_', '')} {acc(v, by, 'P1'):.2f}/{acc(v, by, 'P2'):.2f} (n={len(v)})"
        for k, v in sorted(per.items())))
    out.append("")
    return "\n".join(out), {"g1_pass": g1_pass, "g1_clear": g1_clear, **{k: v[0] for k, v in g.items()}}


if __name__ == "__main__":
    tags = sys.argv[1:] or ["qwen25vl3b", "qwen3vl8b"]
    for t in tags:
        if os.path.exists(os.path.join(RES, f"{t}.jsonl")):
            print(report(t)[0])
