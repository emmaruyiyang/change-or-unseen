"""Write/evict policies for a fixed frame budget K. All are streaming: they see frames in order, once.

Each policy exposes run(times, E, upto) -> list of retained entries (dicts with at least "t", "i"),
newest frame always kept. Baselines follow the families EgoStream compares (sliding window,
uniform keep, similarity-based pruning/merging); StaMem adds state awareness.

StaMem v0 (training-free). Every memory entry is one keyframe of one "view" plus a tag:
  t          time of the stored keyframe (the newest frame is always kept visible)
  since      first time this state was seen
  confirmed  last time a frame showed the same state
  view       id of the view (place/object) the frame shows
  current    False once a later frame of the same view showed a different state (kept as history)
On each new frame, compare its CLIP embedding with the current entries:
  sim >= tau_same          KEEP:   nothing changed; extend the entry's `confirmed` time, no new slot
  tau_view <= sim < same   UPDATE: same view, new state; write it, mark the old entry as history
  sim < tau_view           ADD:    new view
When over budget, evict history before current entries; among the same kind, evict the most
redundant (highest similarity to another kept entry), merging its `confirmed` into its twin.
"""
import numpy as np


def _new(i, t, view, current=True):
    return {"i": i, "t": t, "since": t, "confirmed": t, "view": view, "current": current}


class FIFO:
    name = "fifo"

    def __init__(self, K):
        self.K = K

    def run(self, times, E, upto):
        idx = [i for i, t in enumerate(times) if t <= upto][-self.K:]
        return [_new(i, times[i], i) for i in idx]


class Uniform:
    """Keep K frames spread over [0, now]: when full, drop the frame whose two gaps sum smallest."""
    name = "uniform"

    def __init__(self, K):
        self.K = K

    def run(self, times, E, upto):
        keep = []
        for i, t in enumerate(times):
            if t > upto:
                break
            keep.append(i)
            if len(keep) > self.K:
                gaps = [(times[keep[j + 1]] - times[keep[j - 1]], j) for j in range(1, len(keep) - 1)]
                keep.pop(min(gaps)[1])
        return [_new(i, times[i], i) for i in keep]


class Dedup:
    """Similarity pruning: when full, drop the frame most similar to its temporal neighbour."""
    name = "dedup"

    def __init__(self, K):
        self.K = K

    def run(self, times, E, upto):
        keep = []
        for i, t in enumerate(times):
            if t > upto:
                break
            keep.append(i)
            if len(keep) > self.K:
                sims = [(float(E[keep[j]] @ E[keep[j + 1]]), j) for j in range(len(keep) - 1)]
                keep.pop(max(sims)[1])  # drop the older frame of the most similar adjacent pair
        return [_new(i, times[i], i) for i in keep]


class StaMem:
    name = "stamem"

    def __init__(self, K, tau_same=0.93, tau_view=0.80, hist_first=True, gap_w=0.0, refresh=False):
        self.K, self.tau_same, self.tau_view = K, tau_same, tau_view
        self.hist_first, self.gap_w, self.refresh = hist_first, gap_w, refresh
        self.log = []

    def run(self, times, E, upto):
        mem, n_views, self.log = [], 0, []
        for i, t in enumerate(times):
            if t > upto:
                break
            cur = [m for m in mem if m["current"]]
            if cur:
                sims = [float(E[i] @ E[m["i"]]) for m in cur]
                j = int(np.argmax(sims))
                s, best = sims[j], cur[j]
            else:
                s, best = -1.0, None
            if best is not None and s >= self.tau_same:
                # same state: store the newer equivalent frame in the same slot, keep `since`
                self.log.append((t, "keep", best["t"]))
                best["confirmed"] = t
                if self.refresh or i == len(times) - 1 or times[i + 1] > upto:
                    best["i"], best["t"] = i, t  # newest frame always stays visible
                continue
            if best is not None and s >= self.tau_view:
                best["current"] = False
                best["replaced_at"] = t
                mem.append(_new(i, t, best["view"]))
                self.log.append((t, "update", best["t"]))
            else:
                mem.append(_new(i, t, n_views))
                n_views += 1
                self.log.append((t, "add", None))
            while len(mem) > self.K:
                self._evict(mem, E)
        return sorted(mem, key=lambda m: m["t"])

    def _evict(self, mem, E):
        newest = max(mem, key=lambda m: m["t"])
        cand = [m for m in mem if m is not newest]
        hist = [m for m in cand if not m["current"]]
        pool = (hist or cand) if self.hist_first else cand
        ts = sorted(m["t"] for m in mem)
        span = max(ts[-1] - ts[0], 1e-6)
        best, twin_of = None, None
        for m in pool:
            others = [o for o in mem if o is not m]
            sims = [float(E[m["i"]] @ E[o["i"]]) for o in others]
            k = int(np.argmax(sims))
            before = max([o["t"] for o in others if o["t"] < m["t"]], default=ts[0])
            after = min([o["t"] for o in others if o["t"] > m["t"]], default=ts[-1])
            gap = (after - before) / span  # the hole in time this eviction would leave
            key = (sims[k] - self.gap_w * gap, -m["t"])  # most redundant first; ties: oldest
            if best is None or key > best[0]:
                best, twin_of = (key, m), others[k]
        victim = best[1]
        if victim["current"]:
            twin_of["confirmed"] = max(twin_of["confirmed"], victim["confirmed"])
        mem.remove(victim)


def make(name, K, **kw):
    return {"fifo": FIFO, "uniform": Uniform, "dedup": Dedup, "stamem": StaMem}[name](K, **kw)
