# Belief memory on the STORM-Bench sample episodes: first results

2026-11-05 · Method: [`proposal/method_v2.md`](../../proposal/method_v2.md) · Code: [`evidence.py`](evidence.py), [`memory.py`](memory.py), [`fit_eval.py`](fit_eval.py) · Run on Colab (free T4) with [`colab_run.ipynb`](colab_run.ipynb). Cache and per-question predictions are on Google Drive in `change-or-unseen-colab/cache` and `results`.

## Setup

- **Backbone:** Qwen3-VL-8B-Instruct, frozen, 4-bit NF4 with fp16 compute on a T4. The pilot used MLX 4-bit.
- **Data:** the 7 STORM-Bench example episodes, 71 questions. 50 are real video (39 Known, 11 Uncertain) and 21 are simulated.
- **Evidence:** every 4 s of 1 fps frames plus one causal anchor frame. The VLM scores each candidate value and "not shown" from its first-token distribution. The fact to track is assumed to be in memory from t = 0, which follows the proposal's assumption. The abstention options are never shown to the model while it streams.
- **Belief memory:** drift, then a reliability-weighted evidence update, then a determinability head. It has 13 learned parameters. A value is reported as undetermined when d < 0.5. The model inspects when |d − 0.5| < 0.15, at most once, using a 768 px re-read.
- **Splits:**
  - *Sim→real:* fit on the 21 simulated questions, test on the 50 real ones. This is the proposal's protocol at toy scale.
  - *LOEO:* leave one episode out, 7 folds, with predictions pooled.
- **Metrics:** STORM-BR uses joint correctness, meaning both the answer and the Known/Uncertain status must be right. All numbers are on the 50 real questions.

## Results (real video, n = 50: 39 Known, 11 Uncertain)

### Split 1: fit on simulation, test on real video

| system | acc Known | false abstention (Known) | abstains on Uncertain | BR 2-cell [95% CI] | BR 6-cell | AUROC (1−d) | VLM calls / q |
|---|---|---|---|---|---|---|---|
| Baseline: latest observation | 0.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.20 | – | 4.2 |
| Baseline: P1 full context (STORM protocol) | **0.77** | 0.00 | 0.09 | 0.26 [0.11, 0.46] | 0.29 | – | 1.0 |
| Baseline: P1 + confidence threshold (τ fit on train) | 0.77 | 0.05 | 0.09 | 0.26 [0.11, 0.46] | 0.29 | – | 1.0 |
| Belief memory, initial parameters (no fit) | 0.62 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.21 | – | 5.2 |
| **Belief memory (full)** | 0.62 | 0.13 | **0.45** | **0.51** [0.31, 0.66] | 0.32 | **0.79** | 5.2 |
| – no inspection | 0.62 | 0.13 | 0.45 | 0.51 [0.31, 0.66] | 0.32 | 0.77 | 4.2 |
| – no drift (λ = 0) | 0.59 | 0.10 | 0.45 | 0.51 [0.31, 0.66] | 0.32 | 0.86 | 5.2 |
| – no reliability weight (r = 1) | 0.62 | 0.28 | 0.73 | 0.57 [0.43, 0.70] | 0.48 | 0.77 | 5.2 |
| – d = max belief (no d-head) | 0.62 | 0.05 | 0.27 | 0.41 [0.16, 0.58] | 0.31 | 0.71 | 4.3 |

Fitted parameters: λ = 0.051 per second, likelihood temperature T = 2.47, reliability r = σ(1.94 · logit(vis) − 0.08).

### Split 2: leave one episode out (7 folds, pooled)

| system | acc Known | false abstention (Known) | abstains on Uncertain | BR 2-cell [95% CI] | BR 6-cell | AUROC (1−d) | VLM calls / q |
|---|---|---|---|---|---|---|---|
| Baseline: latest observation | 0.56 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.20 | – | 4.2 |
| Baseline: P1 full context (STORM protocol) | **0.77** | 0.00 | 0.09 | 0.26 [0.11, 0.46] | 0.29 | – | 1.0 |
| Baseline: P1 + confidence threshold (τ fit on train) | 0.77 | 0.36 | **0.64** | **0.57** [0.42, 0.70] | **0.50** | – | 1.0 |
| Belief memory, initial parameters (no fit) | 0.62 | 0.00 | 0.00 | 0.14 [0.10, 0.21] | 0.21 | – | 5.2 |
| **Belief memory (full)** | 0.62 | 0.05 | 0.45 | 0.53 [0.32, 0.68] | 0.33 | 0.68 | 5.2 |
| – no inspection | 0.62 | 0.08 | 0.45 | 0.53 [0.32, 0.68] | 0.33 | 0.74 | 4.2 |
| – no drift (λ = 0) | 0.59 | 0.05 | 0.45 | 0.52 [0.32, 0.67] | 0.33 | 0.64 | 5.2 |
| – no reliability weight (r = 1) | 0.62 | 0.05 | 0.45 | 0.53 [0.32, 0.68] | 0.33 | 0.69 | 5.2 |
| – d = max belief (no d-head) | 0.62 | 0.08 | 0.27 | 0.41 [0.16, 0.58] | 0.31 | 0.65 | 4.3 |

## What the numbers say

1. **The belief memory learns when to abstain, and that transfers from simulation to real video.** Fitted only on the 21 simulated questions, it abstains on 5 of 11 real Uncertain questions. The STORM protocol (P1) abstains on 1 of 11. False abstention on Known questions rises to 5 of 39. BR roughly doubles, from 0.26 to 0.51. Determinability separates Known from Uncertain with AUROC 0.79. With 11 Uncertain items, the confidence intervals overlap.
2. **The same transfer does not hold for a plain confidence threshold.** A threshold on P1's answer probability, tuned on simulation, never fires on real video (BR 0.26). Tuned on other real episodes (LOEO), it reaches BR 0.57, but it abstains on 36% of Known questions. The belief memory's d gets close to that BR (0.53) with 5% false abstention.
3. **The determinability head matters; the other components barely register at this size.** Replacing d with max(b) lowers BR from 0.51 to 0.41 and AUROC from 0.79 to 0.71. Removing drift, removing the reliability weight, or removing inspection changes BR by at most 0.06, which is within the noise for 11 Uncertain items. Inspection rarely moves d across the threshold.
4. **The answers themselves are worse than full context: 0.62 vs 0.77 on Known.** This is the pilot's G2 result again. STORM episodes fit in the VLM context, so reading the whole prefix at once beats accumulating clip-level evidence. Clip-level evidence is a poor fit for counting, history and temporal-order questions. It does beat trusting the latest observation (0.56).

## Obvious next step

Combine the two: take the answer from P1 (full context) and the abstention decision from the belief memory's d. On STORM's short episodes, this should keep 0.77 on Known questions and the higher abstention rate. It needs no new VLM calls, because everything is already in the cache. On longer streams, the bounded-memory setting (method_v2 §7, Setting B) is where the belief has to carry the answer as well.

## Caveats

- **Tiny sample.** There are only 11 Uncertain real questions, so a single question moves "abstains on Uncertain" by 0.09. None of the differences between ablations are reliable.
- **P1 frame cap.** The first 33 questions were extracted with P1 capped at 32 frames. The rest were capped at 16, because a 32-frame prompt (about 9k tokens) ran out of memory in the T4 text attention. Only questions whose query time is over 16 s are affected.
- **Quantization.** 4-bit NF4 is used here, while the pilot used MLX 4-bit. P1 accuracy on Known questions matches the pilot (0.77).
- **The fact to track is given from t = 0.** The question stem and its concrete values are known during streaming, which is the proposal's assumption. The STORM protocol only reveals the question at query time.
- **Matched-pair ranking loss.** It is not used, because the sample has no matched episodes.
