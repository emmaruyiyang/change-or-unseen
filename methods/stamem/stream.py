"""The stream side of StaMem: STORM episodes as 1 fps frames, plus a cached CLIP embedding per frame.

The embedding is the cheap per-frame signal the write policy runs on (no VLM call per frame).
Model: open_clip ViT-B/32 (LAION DataComp-XL), loaded from the local Hugging Face cache.
"""
import glob, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "probes", "storm"))
from probe import duration, frame_path, load_questions  # noqa: E402

CACHE = os.path.join(HERE, "cache")
CLIP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--laion--CLIP-ViT-B-32-DataComp.XL-s13B-b90K/snapshots/*/open_clip_model.safetensors")


def episodes():
    """{episode_id: {"video", "subset", "dur", "times", "questions"}}; times = 1 fps grid over the video."""
    eps = {}
    for q in load_questions():
        e = eps.setdefault(q["episode_id"], {"video": q["video"], "subset": q["subset"], "questions": []})
        e["questions"].append(q)
    for e in eps.values():
        e["dur"] = duration(e["video"])
        e["times"] = [float(k) for k in range(int(math.floor(e["dur"] - 0.05)) + 1)]
    return eps


_clip = None


def _model():
    global _clip
    if _clip is None:
        import open_clip, torch
        m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained=glob.glob(CLIP_GLOB)[0])
        m.eval()
        _clip = (m, pre, torch)
    return _clip


def embed(ep_id, ep):
    """L2-normalised CLIP image embedding per frame of the 1 fps grid, cached to methods/stamem/cache."""
    path = os.path.join(CACHE, f"clip_{ep_id}.npy")
    if os.path.exists(path):
        E = np.load(path)
        if len(E) == len(ep["times"]):
            return E
    from PIL import Image
    m, pre, torch = _model()
    imgs = [pre(Image.open(frame_path(ep["video"], t)).convert("RGB")) for t in ep["times"]]
    with torch.no_grad():
        E = m.encode_image(torch.stack(imgs)).float()
        E = torch.nn.functional.normalize(E, dim=-1).numpy()
    np.save(path, E)
    return E


if __name__ == "__main__":
    for k, e in episodes().items():
        E = embed(k, e)
        s = (E[1:] * E[:-1]).sum(1)
        full = E @ E.T
        print(f"{k[:40]:<40} {e['subset']:<14} n={len(E):>3}  consecutive sim: "
              f"min {s.min():.3f} p10 {np.percentile(s, 10):.3f} med {np.median(s):.3f}  "
              f"all-pairs med {np.median(full[np.triu_indices(len(E), 1)]):.3f}")
