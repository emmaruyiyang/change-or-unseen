"""Answerability (Known / Uncertain) probes, scored on both sides.

S1  STORM's own status probe: online prefix frames + question stem (no options) + the
    4-way status menu from STORM-Bench scripts/eval/eval_storm_diagnostics.py.
S2  Describe-then-judge: the model first writes a note from the online prefix frames
    (question stem, no options), then judges status from the note alone, text only.
    The same note is also used to answer the multiple-choice question (S2-answer), so
    the pipeline needs no oracle frames.

For every status call we keep P(uncertain) = P(C) + P(D) from the first-token
distribution over the four letters, so abstention can be scored at any threshold.

Usage: python probes/storm/status_probe.py --model <mlx id> --tag <tag> [--subsets ...]
"""
import argparse, json, os, re, time

import probe as P  # same data loading, frame sampling, and option order as the P0-P3 probes

STATUS_MENU = (
    "(A) Directly known: the frames show the answer directly, clearly, and uniquely",
    "(B) Known after reasoning: combining the visible evidence still uniquely determines the answer",
    "(C) Missing observation: key frames are missing, the target is not visible, or evidence is insufficient",
    "(D) Ambiguous evidence: multiple candidates fit the frames, or the evidence conflicts",
)
STATUS_OF = {"A": "known", "B": "known", "C": "uncertain", "D": "uncertain"}


def status_question(stem, source="the frames seen so far"):
    """Verbatim STORM wording; `source` is swapped to 'the notes' for S2."""
    menu = "\n".join(STATUS_MENU)
    if source != "the frames seen so far":
        menu = menu.replace("the frames", "the notes").replace("key frames are missing", "key observations are missing")
    return (f"Question: {stem}\n\n"
            "Judge the epistemic status of the evidence needed to answer this question, "
            f"based only on {source}. "
            "Do not answer the original question. Do not fill in off-screen information from common sense. "
            "Choose one of the following and output only the letter:\n" + menu)


class ScoringVLM(P.VLM):
    def __init__(self, model_id):
        super().__init__(model_id)
        tok = getattr(self.processor, "tokenizer", self.processor)
        self.letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}

    def letter_probs(self, frames, times, text):
        """Greedy first token plus its distribution restricted to A-D."""
        import mlx.core as mx
        from mlx_vlm import stream_generate
        prompt = self._prompt(times, text)
        first = next(iter(stream_generate(self.model, self.processor, prompt, image=frames or None,
                                          max_tokens=1, temperature=0.0)))
        lp = first.logprobs.reshape(-1)
        raw = mx.array([lp[self.letter_ids[L]] for L in "ABCD"])
        probs = mx.softmax(raw).tolist()
        mx.clear_cache()
        return first.text, dict(zip("ABCD", probs))

    def _prompt(self, times, text):
        content = []
        for t in times:
            content += [{"type": "text", "text": f"Frame at t={t:.0f}s:"}, {"type": "image"}]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        tok = getattr(self.processor, "tokenizer", self.processor)
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run(args):
    qs = P.load_questions()
    if args.subsets:
        qs = [q for q in qs if q["subset"] in args.subsets]
    out_path = os.path.join(os.path.dirname(__file__), "results", f"status_{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {r["id"] for r in map(json.loads, open(out_path))}
    vlm = ScoringVLM(args.model)
    durs = {}
    with open(out_path, "a") as fo:
        for n, q in enumerate(qs):
            if q["id"] in done:
                continue
            t0 = time.time()
            dur = durs.setdefault(q["video"], P.duration(q["video"]))
            perm = P.permutation(q["id"])
            gold = P.LETTERS[perm.index(q["answer_index"])]
            stem_mcq = f"Question: {q['question']}\nOptions:\n{P.options_block(q, perm)}\n{P.MCQ_TAIL}"
            now = f"The current time is t={q['query_time']:g}s."
            times = P.online_times(q, dur)
            frames = [P.frame_path(q["video"], t) for t in times]

            s1_raw, s1_p = vlm.letter_probs(
                frames, times, f"These are the frames of a video stream seen so far, one per second. {now}\n"
                               + status_question(q["question"]))
            note = vlm.ask(frames, times,
                           f"These are the frames of a video stream seen so far, one per second. {now}\n"
                           f"Someone will later ask: \"{q['question']}\"\n"
                           "Write down, in at most three sentences, what these frames actually show that bears on it: "
                           "objects, locations, states, counts and when they were seen. "
                           "Say explicitly what was not visible or could not be made out.", 160).strip()
            s2_raw, s2_p = vlm.letter_probs([], [], f"Notes taken while watching a video stream ({now}):\n{note}\n\n"
                                                    + status_question(q["question"], "the notes"))
            ans_raw = vlm.ask([], [], f"Notes taken while watching a video stream ({now}):\n{note}\n"
                                      f"Using only these notes, answer.\n{stem_mcq}", 8)
            abst = [P.LETTERS[perm.index(i)] for i, o in enumerate(q["options"])
                    if re.search(r"cannot be determined|uncertain|unresolved|no definite|not uniquely|"
                                 r"no concrete alternative", o, re.I)]
            rec = {"id": q["id"], "subset": q["subset"], "type": q["question_type"],
                   "status": q["diagnostics"]["epistemic_status"], "intensity": q["change_intensity"],
                   "gold": gold, "abstain_letters": abst,
                   "s1_letter": P.parse_letter(s1_raw), "s1_p": s1_p, "s1_p_unc": s1_p["C"] + s1_p["D"],
                   "s2_letter": P.parse_letter(s2_raw), "s2_p": s2_p, "s2_p_unc": s2_p["C"] + s2_p["D"],
                   "s2_answer": P.parse_letter(ans_raw), "note": note, "n_frames": len(times),
                   "sec": round(time.time() - t0, 1)}
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"[{n+1}/{len(qs)}] {q['subset']:<14} gt={rec['status']:<9} "
                  f"S1 p_unc={rec['s1_p_unc']:.2f}  S2 p_unc={rec['s2_p_unc']:.2f}  {rec['sec']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subsets", nargs="+", default=None)
    run(ap.parse_args())
