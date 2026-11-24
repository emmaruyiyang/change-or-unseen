"""Stage 2 of the belief memory (method_v2 §2, §4, §5): the learned update and the decision read-out.

One memory entry per tracked fact holds
  b      belief over the concrete candidate values (torch vector)
  d      determinability: P(the current value is uniquely determined by what was observed)
  t_f    time of the last supporting observation
  anchor time of the anchor frame (chosen in evidence.py)
The update is Bayesian in structure, with a handful of learned parameters:
  drift      b- = (1 - h) b + h u,   h = 1 - exp(-lam * dt)
  evidence   b+ ∝ b- * l^r,          l = softmax(logp_values / T),  r = sigmoid(a * logit(vis) + c)
  d-head     logistic regression on streaming summary features (a GRU is the plan for the full
             simulated training set; with 71 sample questions a GRU would just memorise them)
The four decisions are read out of (b, d); they are not trained as labels.
"""
import math
from dataclasses import dataclass, field

import torch

EPS = 1e-6
N_FEAT = 8


@dataclass
class Config:
    use_drift: bool = True
    use_reliability: bool = True
    use_dhead: bool = True         # False: d = max(b) (a plain confidence rule on the belief)
    tau_d: float = 0.5             # Undetermined if d < tau_d
    tau_u: float = 0.1             # Update if the top value changes and KL(b+ || b-) > tau_u
    inspect_band: float = 0.15     # Inspect if |d - tau_d| < band (budget: one per decision)
    use_inspect: bool = True


class BeliefMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_lam = torch.nn.Parameter(torch.tensor(math.log(0.02)))   # per second
        self.log_T = torch.nn.Parameter(torch.tensor(0.0))
        self.rel_a = torch.nn.Parameter(torch.tensor(2.0))
        self.rel_c = torch.nn.Parameter(torch.tensor(0.0))
        self.d_w = torch.nn.Parameter(torch.zeros(N_FEAT))
        self.d_b = torch.nn.Parameter(torch.tensor(0.0))
        self.init_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

    def prior_penalty(self):
        return sum(((p - self.init_state[k]) ** 2).sum() for k, p in self.named_parameters())

    # ---- one evidence step --------------------------------------------------------------
    def step(self, b, logp, dt, cfg):
        """logp: log-probs over [values..., not-shown]. Returns (b-, b+, vis)."""
        p = torch.softmax(logp, 0)
        vis = (1 - p[-1]).clamp(EPS, 1 - EPS)
        if cfg.use_drift:
            h = 1 - torch.exp(-torch.exp(self.log_lam) * dt)
            b_minus = (1 - h) * b + h / b.numel()
        else:
            b_minus = b
        lik = torch.log_softmax(logp[:-1] / torch.exp(self.log_T), 0)
        r = torch.sigmoid(self.rel_a * torch.logit(vis) + self.rel_c) if cfg.use_reliability else torch.tensor(1.0)
        b_plus = torch.softmax(torch.log(b_minus + EPS) + r * lik, 0)
        return b_minus, b_plus, vis

    def features(self, b, vis_hist, t_last_vis, t_now, flips):
        k = b.numel()
        ent = -(b * torch.log(b + EPS)).sum() / math.log(k)
        v = torch.stack(vis_hist)
        since = math.log1p(max(0.0, t_now - t_last_vis)) / math.log(61) if t_last_vis is not None else 1.0
        return torch.stack([b.max(), 1 - ent, torch.tensor(since), v.mean(), v.max(), v[-1],
                            (v > 0.5).float().mean(), torch.tensor(min(flips, 4) / 4)])

    def determinability(self, b, feats, cfg):
        if not cfg.use_dhead:
            return b.max()
        return torch.sigmoid(self.d_w @ feats + self.d_b)

    # ---- one question, streamed --------------------------------------------------------
    def run(self, rec, cfg, inspect_ok=True, train=False):
        """Stream the cached clips of one question. Returns a dict with the final belief, d,
        the decision trace and how many VLM calls were spent. train=True skips the trace."""
        if "_lp" not in rec:  # build the evidence tensors once per record
            rec["_lp"] = [torch.tensor(list(c["logp"].values())) for c in rec["clips"]]
            rec["_lp_inspect"] = torch.tensor(list(rec["inspect"]["logp"].values()))
        k = len(rec["values"])
        b = torch.full((k,), 1.0 / k)
        vis_hist, t_last_vis, flips, trace = [], None, 0, []
        prev_t = 0.0
        for c, logp in zip(rec["clips"], rec["_lp"]):
            dt = c["t1"] - prev_t
            prev_t = c["t1"]
            b_minus, b_plus, vis = self.step(b, logp, dt, cfg)
            with torch.no_grad():
                kl = float((b_plus * (torch.log(b_plus + EPS) - torch.log(b_minus + EPS))).sum())
                top_changed = int(b_plus.argmax()) != int(b.argmax())
                v = float(vis)
            event = "update" if top_changed and kl > cfg.tau_u else "keep"
            if event == "update":
                flips += 1
            if v > 0.5:
                t_last_vis = c["t1"]
            vis_hist.append(vis)
            b = b_plus
            if not train:
                trace.append({"t": c["t1"], "event": event, "top": int(b.argmax()), "vis": round(v, 3)})
        t_now = rec["query_time"]
        d = self.determinability(b, self.features(b, vis_hist, t_last_vis, t_now, flips), cfg)
        calls, inspected = len(rec["clips"]), False
        if cfg.use_inspect and inspect_ok and abs(d.item() - cfg.tau_d) < cfg.inspect_band:
            logp = rec["_lp_inspect"]
            _, b, vis = self.step(b, logp, 0.0, cfg)
            vis_hist.append(vis)
            if vis.item() > 0.5:
                t_last_vis = t_now
            d = self.determinability(b, self.features(b, vis_hist, t_last_vis, t_now, flips), cfg)
            calls, inspected = calls + 1, True
        return {"b": b, "d": d, "trace": trace, "calls": calls, "inspected": inspected}


def pack(recs):
    """Pad a list of cached questions into batch tensors for BeliefMemory.run_batch (training only)."""
    Q, T = len(recs), max(len(r["clips"]) for r in recs)
    L = torch.full((Q, T, 5), -1e9)
    mask = torch.zeros(Q, T, dtype=torch.bool)
    dt = torch.zeros(Q, T)
    t1 = torch.zeros(Q, T)
    vmask = torch.zeros(Q, 4, dtype=torch.bool)
    for i, r in enumerate(recs):
        k = len(r["values"])
        vmask[i, :k] = True
        prev = 0.0
        for j, c in enumerate(r["clips"]):
            lp = list(c["logp"].values())
            L[i, j, :k] = torch.tensor(lp[:k])
            L[i, j, 4] = lp[k]
            mask[i, j] = True
            dt[i, j] = c["t1"] - prev
            t1[i, j] = c["t1"]
            prev = c["t1"]
    tq = torch.tensor([float(r["query_time"]) for r in recs])
    return {"L": L, "mask": mask, "dt": dt, "t1": t1, "vmask": vmask, "tq": tq,
            "k": vmask.sum(1).float()}


def run_batch(m, P, cfg):
    """Vectorised version of BeliefMemory.run without inspection; returns (b [Q,4], d [Q]).
    Must match run() exactly; checked against it in fit_eval.selfcheck()."""
    L, mask, vmask, k = P["L"], P["mask"], P["vmask"], P["k"]
    Q, T, _ = L.shape
    NEG = torch.tensor(-1e9)
    b = vmask.float() / k[:, None]
    vsum = torch.zeros(Q); vmax = torch.zeros(Q); vlast = torch.zeros(Q); vcnt = torch.zeros(Q)
    n = torch.zeros(Q); flips = torch.zeros(Q); t_last = torch.full((Q,), -1.0)
    for t in range(T):
        act = mask[:, t]
        if not act.any():
            break
        lt = L[:, t]
        p = torch.softmax(lt, -1)
        vis = (1 - p[:, 4]).clamp(EPS, 1 - EPS)
        if cfg.use_drift:
            h = 1 - torch.exp(-torch.exp(m.log_lam) * P["dt"][:, t])
            b_minus = (1 - h)[:, None] * b + (h / k)[:, None] * vmask.float()
        else:
            b_minus = b
        lik = torch.log_softmax(torch.where(vmask, lt[:, :4] / torch.exp(m.log_T), NEG), -1)
        r = torch.sigmoid(m.rel_a * torch.logit(vis) + m.rel_c) if cfg.use_reliability else torch.ones(Q)
        logits = torch.where(vmask, torch.log(b_minus + EPS) + r[:, None] * lik, NEG)
        b_plus = torch.softmax(logits, -1)
        with torch.no_grad():
            kl = (b_plus * (torch.log(b_plus + EPS) - torch.log(b_minus + EPS))).sum(-1)
            upd = (b_plus.argmax(-1) != b.argmax(-1)) & (kl > cfg.tau_u) & act
        flips = flips + upd.float()
        b = torch.where(act[:, None], b_plus, b)
        vd = vis.detach()
        t_last = torch.where(act & (vd > 0.5), P["t1"][:, t], t_last)
        vsum = vsum + torch.where(act, vis, torch.zeros(Q))
        vmax = torch.where(act, torch.maximum(vmax, vis), vmax)
        vlast = torch.where(act, vis, vlast)
        vcnt = vcnt + (act & (vd > 0.5)).float()
        n = n + act.float()
    ent = -(b * torch.log(b + EPS)).sum(-1) / torch.log(k)
    since = torch.where(t_last >= 0, torch.log1p((P["tq"] - t_last).clamp(min=0)) / math.log(61), torch.ones(Q))
    feats = torch.stack([b.max(-1).values, 1 - ent, since, vsum / n, vmax, vlast, vcnt / n,
                         flips.clamp(max=4) / 4], -1)
    d = torch.sigmoid(feats @ m.d_w + m.d_b) if cfg.use_dhead else b.max(-1).values
    return b, d


def decide(rec, out, cfg):
    """Map (b, d) to STORM's answer + status. Returns (option index, predicted status)."""
    if out["d"].item() < cfg.tau_d:
        ans = rec["abstain"][0] if rec["abstain"] else rec["values"][int(out["b"].argmax())]
        return ans, "uncertain"
    return rec["values"][int(out["b"].argmax())], "known"
