# Perception go/no-go: pilot results

2026-11-04 · Decision rules fixed in advance in [PREREGISTRATION.md](PREREGISTRATION.md) · Code: [probe.py](probe.py), [analyze.py](analyze.py) · Raw outputs: `results/*.jsonl`

## Verdict: conditional GO, with a backbone switch and a change of framing

1. **Perception is good enough with Qwen3-VL-8B (G1: point estimate passes, interval straddles the threshold).** With the right frames in hand, the 8B model answers 77% of real-video Known questions, against 36% with no video. The 3B model reaches 74%, but loses more when its observations are written down as text (see 3). Use 8B or larger as the main backbone and keep 3B as a small-model ablation.
2. **On Known questions there is no headroom for memory (G2 fails for both models).** Seeing the whole prefix (P1) is as accurate as seeing only the annotated evidence frames (P2): a gap of 0.00 for 8B and +0.03 for 3B. STORM episodes are short (8–43 s here, at most 32 frames), so the whole history fits in the VLM's context and there is nothing to forget or retrieve. **A controller cannot gain anything on Known questions on STORM as it stands.**
3. **Text-only memory is not enough (G3 fails for both models).** Turning the evidence frames into a written note before answering costs 15 points (8B) and 21 points (3B). The notes invent details (a "glue" bottle that was lube, "blue" liquid that was red, a tripod count that rises frame by frame) and drop the fine attributes that state questions hinge on (closed → open). This supports the hybrid memory design: a structured slot plus a visual anchor.
4. **The headroom is in abstention.** On real Uncertain questions, both models pick the "cannot be determined" option only 2 times out of 11 when they see the frames together with the options (P1). After the describe-first pipeline (P3), they abstain 7/11 (3B) and 6/11 (8B). Looking at images and options together makes the model overconfident; writing out what it actually saw first makes it more cautious. *Caveat:* only 1 of the 39 Known items offers an abstention option, so this pilot cannot measure false abstention. P3 may just be abstaining across the board.

## Numbers (real video, Known questions, n = 39; 95% bootstrap intervals)

| probe | Qwen2.5-VL-3B | Qwen3-VL-8B (4-bit) |
|---|---|---|
| P0 blind (options only) | 0.44 [0.28, 0.59] | 0.36 [0.21, 0.51] |
| P1 online, full prefix | 0.72 [0.56, 0.85] | 0.77 [0.64, 0.90] |
| P2 oracle evidence frames | 0.74 [0.62, 0.87] | 0.77 [0.64, 0.90] |
| P3 extract note → answer | 0.54 [0.38, 0.69] | 0.62 [0.46, 0.77] |

In-scope types only (current_state, object_tracking, state_change; n = 22), P1 / P2 / P3: 3B 0.68 / 0.77 / 0.50; 8B 0.82 / 0.82 / 0.68.

Per domain, 8B P1 / P2: bike 1.00/0.90 (n=10), cook 0.88/1.00 (n=8), health 0.50/0.50 (n=8), music 0.83/0.83 (n=6), sports 0.57/0.57 (n=7). Health and Sports are weak for both models, so perception, not memory, limits those domains.

Real Uncertain questions (n = 11), share of answers that choose the abstention option:

| probe | 3B | 8B |
|---|---|---|
| P0 blind | 9/11 | 4/11 |
| P1 online | 2/11 | 2/11 |
| P2 oracle evidence | 2/11 | 2/11 |
| P3 extract → answer | 7/11 | 6/11 |

## What this means for the proposal

- **Change the backbone** from Qwen2.5-VL-3B to Qwen3-VL-8B (or larger, on a GPU machine).
- **Move the headline claim from fact recovery to epistemic decisions.** On STORM, the controller has to win by knowing when it does not know (STORM-BR, Uncertain questions), not by recovering forgotten facts. This fits the probabilistic-belief design, and P3's behaviour suggests a cheap baseline to beat: describe first, then decide.
- **If memory itself is to be a contribution, add a setting where memory binds.** Give the controller a fixed budget of K memory slots or frames, smaller than the episode, or chain episodes into longer streams. The proposal's Discussion already mentions a fixed number of memory entries; promote it to a main setting.
- **Store visual evidence in memory, not only text.** Report text-only versus hybrid as an ablation.
- **Next measurement:** run STORM's own answerability probe (question without options) on Known and Uncertain together, so abstention is scored on both sides. That tells us whether P3 is better calibrated or simply more timid.

## Limits of this pilot

- Only the 7 example episodes from the STORM-Bench repo were available: the Hugging Face video release returns 404, and the full real-video set cannot be rebuilt from HD-EPIC / Ego-Exo4D because clip boundaries were removed. All intervals are wide, and every G1 quantity except 8B's P2 − P0 straddles its threshold. **Rerun with `probe.py` on the full release when it becomes available.**
- Deviations from the pre-registration: the simulation episodes were dropped from the 8B run, and the 3B run stopped after 64/71 questions, because a Metal memory-cache leak pushed the Mac into swap (since fixed in `probe.py`). The primary real-video set is complete for both models. The 8B model is 4-bit quantized. Options are shuffled with our own id-derived permutation, not STORM's exact one.
- Every question is multiple choice, so P0 is not 25% by chance: language priors and option wording leak information (3B blind accuracy is 0.44).
