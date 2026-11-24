"""Stage 1 of the belief memory (method_v2 §3): cache frozen-VLM evidence for STORM sample episodes.

For every question we assume the fact to track is already in memory (proposal §Discussion), so the
question stem and its concrete candidate values are known from t = 0. The answer options that mean
"cannot be determined" are NOT shown as values; the model never sees them during streaming.

Three kinds of VLM calls, all scored from the first-token distribution (no generation):
  clip     every W seconds of 1 fps frames up to query time, plus the memory anchor frame:
           P(value_k) for each candidate value and P(not shown) -> likelihood + visibility.
  p1       STORM online protocol (<= 32 frames, all four options): full-context baseline.
  inspect  one high-resolution re-read (768 px) of the clearest clip and the newest clip.
           Computed for every question so the offline evaluation can charge it only when the
           policy asks for it (same output as calling it online).

The anchor frame is chosen causally and without any learned parameter: the middle frame of the
most recent earlier clip whose visibility was > 0.5. So the cache does not depend on the fit.

Usage: .venv/bin/python methods/belief/evidence.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit --tag qwen3vl8b --hires
Low-load (the 8B run hard-reset a MacBook Air via the SoC watchdog three times):
  .venv/bin/python methods/belief/evidence.py --model mlx-community/Qwen2.5-VL-3B-Instruct-bf16 --tag qwen25vl3b \
      --p1_frames 16 --mem_gb 8 --pause 2 --limit 1
"""
import argparse, json, math, os, re, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probes", "storm"))
import probe as P  # noqa: E402  same data loading, 1 fps frame cache and option permutation as the pilot

HERE = os.path.dirname(__file__)
W = 4                 # clip length in seconds (frames at 1 fps)
HI_SIDE = 768         # inspection resolution
HI_FRAMES = os.path.join(HERE, "..", "..", "data", "storm", "frames_hi")
LET = "ABCDE"
ABSTAIN_RE = re.compile(r"cannot be determined|uncertain|unresolved|no definite|not uniquely|"
                        r"no concrete alternative", re.I)


def split_options(q):
    """Concrete values (shown while streaming) vs abstention options, in the fixed per-question order."""
    perm = P.permutation(q["id"])
    vals = [i for i in perm if not ABSTAIN_RE.search(q["options"][i])]
    abst = [i for i in perm if ABSTAIN_RE.search(q["options"][i])]
    return vals, abst


def grid(q, dur):
    last = min(q["query_time"], dur - 0.05)
    return [float(k) for k in range(int(math.floor(last)) + 1)]


def clips(q, dur):
    g = grid(q, dur)
    return [g[i:i + W] for i in range(0, len(g), W)]


def hi_frame(video, t):
    d = os.path.join(HI_FRAMES, os.path.splitext(os.path.basename(video))[0])
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"t{t:05.1f}.jpg")
    if not os.path.exists(p):
        vf = f"scale='if(gt(iw,ih),{HI_SIDE},-2)':'if(gt(iw,ih),-2,{HI_SIDE})'"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                        "-vf", vf, "-q:v", "2", "-y", p], check=True)
    return p


class Scorer(P.VLM):
    def __init__(self, model_id):
        super().__init__(model_id)
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in LET}

    def score(self, frames, labels, text, letters):
        """Distribution of the first generated token, restricted to `letters`."""
        import mlx.core as mx
        from mlx_vlm import stream_generate
        content = []
        for lab in labels:
            content += [{"type": "text", "text": lab}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        msgs = [{"role": "user", "content": content}]
        tok = getattr(self.processor, "tokenizer", self.processor)
        try:
            prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        first = next(iter(stream_generate(self.model, self.processor, prompt, image=frames or None,
                                          max_tokens=1, temperature=0.0)))
        lp = first.logprobs.reshape(-1)
        raw = [lp[self.ids[L]].item() for L in letters]
        mx.clear_cache()
        self.peak = max(getattr(self, "peak", 0), mx.get_peak_memory())
        if PAUSE:
            time.sleep(PAUSE)  # let the GPU idle between calls (the 8B run reset the Mac via watchdog)
        return dict(zip(letters, raw))  # log-probs; normalised later


class HFScorer:
    """Same interface as Scorer, on CUDA with transformers (for Colab). bf16 if the GPU has >= 22 GB
    and supports it, otherwise 4-bit NF4 via bitsandbytes."""

    def __init__(self, model_id, quant="auto"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        prop = torch.cuda.get_device_properties(0)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16  # T4: fp16
        kw = {"dtype": dtype, "device_map": "cuda"}
        if quant == "4bit" or (quant == "auto" and prop.total_memory < 22 * 2**30):
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype)
        print(f"GPU {prop.name} {prop.total_memory / 2**30:.0f} GB, dtype {dtype}, "
              f"{'4-bit' if 'quantization_config' in kw else 'full precision'}", flush=True)
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        tok = self.processor.tokenizer
        self.ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in LET}
        self.peak = 0

    def score(self, frames, labels, text, letters):
        from PIL import Image
        torch = self.torch
        content = []
        for lab in labels:
            content += [{"type": "text", "text": lab}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(f).convert("RGB") for f in frames]
        inputs = self.processor(text=[prompt], images=imgs or None, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()  # full-vocab logits for every position OOMs a T4
        lp = torch.log_softmax(logits, -1)
        self.peak = max(self.peak, torch.cuda.max_memory_allocated())
        return {L: lp[self.ids[L]].item() for L in letters}


def value_menu(q, vals):
    k = len(vals)
    lines = [f"{LET[j]}. {q['options'][i]}" for j, i in enumerate(vals)]
    lines.append(f"{LET[k]}. These frames do not show enough to tell.")
    return "\n".join(lines), LET[:k + 1]


def clip_prompt(q, vals, t_end, has_anchor):
    menu, letters = value_menu(q, vals)
    anchor = ("The first image is a memory frame: the last earlier moment where the subject was clearly visible. "
              "The other images are new frames. " if has_anchor else "")
    return (f"You are watching a video stream and tracking this question: \"{q['question']}\"\n"
            f"{anchor}Using only what is visible in these images, which option holds as of t={t_end:.0f}s? "
            "Do not guess about anything that is not visible.\n"
            f"{menu}\nAnswer with the letter only."), letters


PAUSE = 0.0


def run(args):
    global PAUSE
    if args.backend == "mlx" and args.mem_gb:
        import mlx.core as mx
        mx.set_memory_limit(int(args.mem_gb * 2**30))
        mx.set_cache_limit(int(args.cache_gb * 2**30))
    PAUSE = args.pause
    P.MAX_FRAMES = args.p1_frames
    qs = P.load_questions()
    if args.subsets:
        qs = [q for q in qs if q["subset"] in args.subsets]
    if args.real_first:
        qs.sort(key=lambda q: q["subset"] in ("storm_sim", "storm_vhome"))
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    out = os.path.join(HERE, "cache", f"{args.tag}.jsonl")
    done = {r["id"] for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    vlm = HFScorer(args.model, args.quant) if args.backend == "hf" else Scorer(args.model)
    durs = {}
    new = 0
    with open(out, "a") as fo:
        for n, q in enumerate(qs):
            if q["id"] in done:
                continue
            if args.limit and new >= args.limit:
                break
            new += 1
            t0 = time.time()
            dur = durs.setdefault(q["video"], P.duration(q["video"]))
            vals, abst = split_options(q)
            rec = {"id": q["id"], "subset": q["subset"], "type": q["question_type"],
                   "status": q["diagnostics"]["epistemic_status"], "intensity": q["change_intensity"],
                   "query_time": q["query_time"], "evidence_spans": q["evidence_spans"],
                   "values": vals, "abstain": abst, "answer_index": q["answer_index"], "clips": []}

            # clip-wise evidence with a causal, parameter-free anchor
            anchor_t = None
            for ts in clips(q, dur):
                labels, frames = [], []
                if anchor_t is not None:
                    labels.append(f"Memory frame at t={anchor_t:.0f}s:")
                    frames.append(P.frame_path(q["video"], anchor_t))
                for t in ts:
                    labels.append(f"Frame at t={t:.0f}s:")
                    frames.append(P.frame_path(q["video"], t))
                text, letters = clip_prompt(q, vals, ts[-1], anchor_t is not None)
                lp = vlm.score(frames, labels, text, letters)
                rec["clips"].append({"t0": ts[0], "t1": ts[-1], "anchor": anchor_t, "logp": lp})
                z = [math.exp(v) for v in lp.values()]
                if z[-1] / sum(z) < 0.5:          # visible in this clip -> its middle frame is the new anchor
                    anchor_t = ts[len(ts) // 2]

            # P1: STORM online protocol with all four options
            perm = P.permutation(q["id"])
            times = P.online_times(q, dur)
            stem = (f"These are the frames of a video stream seen so far, one per second. "
                    f"The current time is t={q['query_time']:g}s.\nQuestion: {q['question']}\n"
                    f"Options:\n{P.options_block(q, perm)}\n{P.MCQ_TAIL}")
            rec["p1"] = {"perm": perm, "logp": vlm.score([P.frame_path(q["video"], t) for t in times],
                                                        [f"Frame at t={t:.0f}s:" for t in times], stem, "ABCD")}

            # inspection: clearest clip + newest clip, high resolution, at most 8 frames
            vis = [1 - math.exp(c["logp"][LET[len(vals)]]) / sum(math.exp(v) for v in c["logp"].values())
                   for c in rec["clips"]]
            best = max(range(len(vis)), key=lambda i: vis[i])
            its = sorted(set(clips(q, dur)[best]) | set(clips(q, dur)[-1]))[-8:]
            text, letters = clip_prompt(q, vals, its[-1], False)
            text = text.replace("Using only what is visible",
                                "Look closely: these are high-resolution re-reads of earlier frames. Using only what is visible")
            rec["inspect"] = {"times": its,
                              "hires": args.hires,
                              "logp": vlm.score([(hi_frame if args.hires else P.frame_path)(q["video"], t) for t in its],
                                                [f"Frame at t={t:.0f}s:" for t in its], text, letters)}
            rec["sec"] = round(time.time() - t0, 1)
            rec["p1_frames"] = len(times)
            fo.write(json.dumps(rec) + "\n"); fo.flush(); os.fsync(fo.fileno())  # survive a hard reset
            print(f"[{n+1}/{len(qs)}] {q['subset']:<14} {rec['status']:<9} clips={len(rec['clips'])} "
                  f"{rec['sec']}s peak={vlm.peak / 2**30:.1f}GB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--real_first", action="store_true")
    ap.add_argument("--backend", choices=["mlx", "hf"], default="mlx", help="hf = transformers on CUDA (Colab)")
    ap.add_argument("--quant", choices=["auto", "4bit", "none"], default="auto", help="hf backend only")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new questions")
    ap.add_argument("--p1_frames", type=int, default=32, help="frame cap for the P1 call (STORM uses 32)")
    ap.add_argument("--hires", action="store_true", help="inspect at 768 px instead of 448 px")
    ap.add_argument("--mem_gb", type=float, default=0, help="MLX memory limit, 0 = default")
    ap.add_argument("--cache_gb", type=float, default=1.0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to idle after each VLM call")
    run(ap.parse_args())
