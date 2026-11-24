"""Long-video episodes for the StaMem headroom check, in the same shape as stream.episodes().

  ovo       OVO-Bench backward questions (EPM/ASI/HLD) asked at t >= 120 s  (data/ovo/long_samples.jsonl)
  ovo_backward  all OVO backward questions at t >= 120 s with a local video (data/ovo/long_backward.jsonl)
  meminstream  MemInStream sample with annotated evidence spans (data/meminstream/sample.jsonl)
  videomme  Video-MME long videos (30-60 min); every question is asked at the end of the video
            (data/videomme/long_samples.jsonl)

Frames: one ffmpeg pass per video at 1 fps (short side scaled so the long side is 448 px), stored as
data/<set>/frames1fps/<video>/t%05.1f.jpg. CLIP embeddings are computed in batches and cached in methods/stamem/cache.
Each episode carries its own frame(t) function so answer.py does not need to know where frames live.
"""
import glob, hashlib, json, math, os, re, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "probes", "storm"))
from probe import MAX_SIDE, duration  # noqa: E402

CACHE = os.path.join(HERE, "cache")


def _frames_dir(name, video):
    return os.path.join(DATA, name, "frames1fps", os.path.splitext(video.replace("/", "__"))[0])


def extract(name, video_path, key, n):
    """All 1 fps frames t = 0..n-1 in one ffmpeg pass; returns the directory."""
    d = _frames_dir(name, key)
    done = os.path.join(d, ".done")
    if os.path.exists(done) or len(glob.glob(os.path.join(d, "t*.jpg"))) >= n:
        return d
    os.makedirs(d, exist_ok=True)
    vf = f"fps=1:round=down,scale='if(gt(iw,ih),{MAX_SIDE},-2)':'if(gt(iw,ih),-2,{MAX_SIDE})'"
    subprocess.run(["ffmpeg", "-v", "error", "-i", video_path, "-vf", vf, "-q:v", "4",
                    "-frames:v", str(n), "-start_number", "0", "-y", os.path.join(d, "f%05d.jpg")], check=True)
    for f in glob.glob(os.path.join(d, "f*.jpg")):
        k = int(re.search(r"f(\d+)\.jpg", f).group(1))
        os.replace(f, os.path.join(d, f"t{float(k):05.1f}.jpg"))
    open(done, "w").close()  # ffmpeg may yield one frame fewer than n; do not re-extract every load
    return d


def _episode(name, key, video_path, questions):
    dur = duration(video_path)
    n = int(math.floor(dur - 0.05)) + 1
    d = extract(name, video_path, key, n)
    have = sorted(float(os.path.basename(f)[1:-4]) for f in glob.glob(os.path.join(d, "t*.jpg")))
    times = [t for t in have if t < n]
    return {"video": video_path, "subset": name, "dur": dur, "times": times, "questions": questions,
            "frame": (lambda t, d=d: os.path.join(d, f"t{t:05.1f}.jpg")), "ep_key": f"{name}_{key}"}


def ovo_episodes(fname="long_samples.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "ovo", fname))]
    by = {}
    for r in rows:
        q = {"id": f"ovo{r['id']}", "query_time": float(r["realtime"]), "question": r["question"],
             "options": r["options"], "answer_index": r["gt"], "question_type": r["task"],
             "status": "na", "evidence_spans": None}
        by.setdefault(r["video"], []).append(q)
    eps = {}
    for v, qs in by.items():
        path = os.path.join(DATA, "ovo", "videos", v)
        if os.path.exists(path):
            e = _episode("ovo", v, path, qs)
            eps[e["ep_key"]] = e
    return eps


def videomme_episodes():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "videomme", "long_samples.jsonl"))]
    by = {}
    for r in rows:
        opts = [re.sub(r"^[A-D]\.\s*", "", o) for o in r["options"]]
        q = {"id": f"vmme{r['question_id']}", "query_time": None, "question": r["question"],
             "options": opts, "answer_index": "ABCD".index(r["answer"]), "question_type": r["task_type"],
             "status": "na", "evidence_spans": None}
        by.setdefault(r["videoID"], []).append(q)
    eps = {}
    for v, qs in by.items():
        path = os.path.join(DATA, "videomme", "videos", v + ".mp4")
        if not os.path.exists(path):
            continue
        e = _episode("videomme", v, path, qs)
        for q in qs:
            q["query_time"] = e["times"][-1]  # asked once the whole video has streamed past
        eps[e["ep_key"]] = e
    return eps


def meminstream_episodes():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "meminstream", "sample.jsonl"))]
    rows = [r for r in rows if isinstance(r["correct_index"], int)]  # 37/3799 have several correct options: skipped
    by = {}
    for r in rows:
        opts = [re.sub(r"^[A-E]\.\s*", "", o) for o in r["options"]]
        q = {"id": "mis" + hashlib.md5((r["video_id"] + r["question"]).encode()).hexdigest()[:10], "query_time": float(r["query_time"]),
             "question": r["question"], "options": opts, "answer_index": r["correct_index"],
             "question_type": r["level2"], "level1": r["level1"], "status": "na", "evidence_spans": r["evidence"]}
        by.setdefault(r["local"], []).append(q)
    eps = {}
    for v, qs in by.items():
        e = _episode("meminstream", v, os.path.join(DATA, "meminstream", "videos", v), qs)
        eps[e["ep_key"]] = e
    return eps


def episodes(which):
    return {"ovo": ovo_episodes, "ovo_backward": lambda: ovo_episodes("long_backward.jsonl"),
            "videomme": videomme_episodes, "meminstream": meminstream_episodes}[which]()


_clip = None


def embed(ep, batch=128):
    path = os.path.join(CACHE, f"clip_{ep['ep_key'].replace('/', '__')}.npy")
    if os.path.exists(path):
        E = np.load(path)
        if len(E) == len(ep["times"]):
            return E
    global _clip
    from PIL import Image
    if _clip is None:
        from stream import _model
        _clip = _model()
    m, pre, torch = _clip
    out = []
    with torch.no_grad():
        for s in range(0, len(ep["times"]), batch):
            imgs = [pre(Image.open(ep["frame"](t)).convert("RGB")) for t in ep["times"][s:s + batch]]
            out.append(torch.nn.functional.normalize(m.encode_image(torch.stack(imgs)).float(), dim=-1).numpy())
    E = np.concatenate(out)
    np.save(path, E)
    return E


if __name__ == "__main__":
    which = sys.argv[1]
    for k, e in episodes(which).items():
        E = embed(e)
        s = (E[1:] * E[:-1]).sum(1)
        print(f"{k[:50]:<50} n={len(E):>5} q={len(e['questions']):>2} consecutive sim med {np.median(s):.3f}",
              flush=True)
