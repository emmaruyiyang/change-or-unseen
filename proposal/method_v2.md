# Revised Technical Approach (v2)

Draft, 2026-11-04. Replaces "Proposed Technical Approach" and "Preliminary Experiments and Evaluation" in the original proposal. Changes are driven by the perception pilot in `gonogo/REPORT.md`.

## What the pilot changed

| Pilot finding (real-video STORM examples, Qwen3-VL-8B) | Consequence for the method |
|---|---|
| Seeing the whole prefix (P1) is as accurate as seeing only the annotated evidence frames (P2): 0.77 vs 0.77. Episodes are 8–43 s and fit in context. | On STORM as released, memory has no room to help with Known questions. Memory must be tested in a **bounded-memory** setting, where the stream is longer than what the agent can keep. |
| Answering from a written note instead of the frames costs 15 points (8B) and 21 points (3B). Notes invent details and drop fine state attributes. | Memory entries store a **visual anchor** next to the structured fact, not text alone. |
| On real Uncertain questions, the model chooses "cannot be determined" only 2/11 times when it sees frames and options together. Describe-then-judge raises this to 6/11. | The main gain should come from **knowing when not to answer**. The controller must output calibrated uncertainty, not only a decision. |
| With the right frames, 3B reaches 0.74 and 8B 0.77 on Known questions. 3B loses more when its observations are turned into text. | Backbone: **Qwen3-VL-8B** (frozen). Qwen2.5-VL-3B is kept as a small-model ablation. |

## 1. Overview

We replace the four-way action classifier from the proposal with a **belief memory**. Each memory entry keeps a probability distribution over the state of one tracked fact. A small learned update module revises that distribution as new clips arrive. Update, keep, inspect and undetermined are no longer labels the model is trained to predict. They are **read out** from the belief by thresholds, so they can be tuned or swept after training.

This has three advantages over the four-way classifier:
1. **No hand-made action labels.** The hardest label in the original design was "inspect," which has no clear ground truth in simulation. The belief only needs labels for the world state and for whether that state can be determined from the frames seen so far. The simulator provides both.
2. **Calibration can be measured directly.** One belief gives a whole risk–coverage curve, AUROC for abstention, and STORM-BR at any operating point.
3. **The same score drives memory management.** The bounded-memory setting needs to decide which entries to keep, and the belief tells us which entries are still informative.

## 2. Memory entry

For a tracked fact *f* (an object and an attribute, such as mug / location or lid / open-closed), the memory holds

  m_f = ( V_f, b_f, d_f, t_f, a_f )

- **V_f**: the candidate values seen so far, as text, for example {table, shelf, sink}. These are proposed by the VLM and extended as new values appear.
- **b_f**: the belief, a distribution over V_f ∪ {other}.
- **d_f ∈ [0, 1]**: determinability, the probability that the current state can be determined from what has been observed.
- **t_f**: the time of the last observation that supported the belief.
- **a_f**: the visual anchor, which is the frozen-VLM feature of the supporting frame plus a pointer to that frame in a small frame buffer. The anchor is what an inspection re-reads.

## 3. Evidence extraction (frozen VLM)

For each new clip (1 fps, the same protocol as the pilot), the frozen VLM produces, for each tracked fact:
- a short **describe-first** note about the fact, generated without seeing any question options (pilot probe P3 / S2), and its text embedding;
- **visual features** pooled over the clip and over the frames where the object is localized;
- a **value likelihood**: the VLM's first-token scores for each candidate value in V_f, given the clip;
- a **visibility score**: P(object and attribute are visible in this clip), read from a yes/no prompt.

All VLM outputs are computed once and cached. Training only touches the small update module, so it does not repeat VLM forward passes.

## 4. Belief update

The update has a fixed Bayesian structure, and its parameters are learned:

1. **Drift while unobserved.** Between observations, the belief moves toward "may have changed":
   b⁻ = (1 − h) · b_f + h · u, with h = 1 − exp(−λ_θ · Δt).
   Here u is a broad distribution over values, and the hazard rate λ_θ is predicted from the object category and the scene context. Objects that people are handling change faster than furniture.
2. **Evidence step.** b⁺ ∝ b⁻ · ℓ_θ(e_t)^{r_θ(e_t)}. Here ℓ_θ is a learned likelihood over values computed from the cached evidence. The reliability weight r_θ ∈ [0, 1] scales down evidence when the object is occluded, small, blurred, or seen under a lighting change.
3. **Determinability.** d_f is updated by a small GRU over the evidence sequence and Δt. It is trained to answer one question: given the frames so far, is the current state uniquely determined?

This structure separates the three cases from the problem statement:
- **The camera turns away:** visibility is low, so r ≈ 0. Only the slow drift applies and the value stays the same.
- **The object moves in view:** visibility is high and the likelihood shifts, so the belief updates.
- **The object changes out of view:** there is no evidence, the drift grows, d_f falls, and the system reports undetermined. It does not report "unchanged."

## 5. Reading out the four decisions

| Decision | Rule (thresholds tuned on held-out simulation, then swept for curves) |
|---|---|
| **Update** | The top value of b_f changes and KL(b⁺ ‖ b⁻) > τ_u |
| **Keep** | No update; the entry keeps its value and its old t_f (it is not re-timestamped) |
| **Undetermined** | d_f < τ_d, or max b_f < τ_p |
| **Inspect** | Expected reduction in answer error from re-reading the anchor interval at higher resolution or with more frames, minus a cost c, is positive. At most one inspection per decision, as in the proposal. |

To answer a question at query time, the model matches each option to the values in V_f by text-embedding similarity, so each option gets a probability under b_f. If the fact is Undetermined, the model abstains: it picks the abstention option when the question has one, and otherwise reports "cannot be determined."

**Ablation: a deterministic controller.** The original four-way classifier is trained on the same features, using rule-derived action labels. It is reported as an ablation, which tests whether the belief formulation itself helps.

## 6. Training from matched simulated episodes

We generate episodes with the public STORM-Sim pipeline (AI2-THOR, VirtualHome), using new scenes and seeds. Each base episode comes in matched variants: an in-view move or state change; a view change or lighting change with no state change; and an occlusion or an out-of-view change. The simulator logs the true state s_t and, for each time step, whether s_t can be determined from the frames up to t.

Losses:
- **Value loss:** cross-entropy between b_f and the true state, applied **only at steps where the state is determinable** from the observed history. This prevents leakage of facts the video never showed.
- **Determinability loss:** binary cross-entropy between d_f and the simulator's determinability label.
- **Matched-pair ranking loss:** the predicted probability that the state changed must be higher in the in-view-change variant than in its view-change twin. In the out-of-view-change variant, d_f must be lower than in the occlusion-without-change twin.
- **Calibration:** temperature scaling on a small real-video calibration split. This split is excluded from evaluation.

Fallback if our own rendering is late: train on released STORM-Sim episodes and evaluate on real video only. Say this clearly in the report.

## 7. Two evaluation settings

**Setting A: short streams (STORM as released).** The whole prefix fits in context, so memory is not expected to help accuracy on Known questions. The measured gains are in abstention and calibration: STORM-BR, overconfidence on Uncertain questions, false abstention on Known questions, and AUROC of 1 − d_f.

**Setting B: bounded memory (the main memory setting).** The agent reads the stream without seeing the question and keeps at most **K** entries plus a frame buffer of **B** frames. K and B are set so that they cannot hold the full stream. Streams come from:
- STORM episodes concatenated into one long stream, with questions about earlier episodes asked later;
- OVO-Bench backward-tracing tasks (a sample is already in `data/ovo`);
- ProactEgoMem state- and location-tracking tasks.

When memory is full, the entry with the lowest retention score is evicted. The score is d_f · (1 − h_f), which favors entries that are still determinable and probably still true. We compare this against FIFO and LRU eviction. We sweep K, so the results show how answer accuracy and abstention change as memory gets tighter.

## 8. Baselines, metrics and ablations

**Baselines** (same frozen VLM, same frames and the same inspection budget):
1. always trust the latest observation;
2. a confidence-threshold rule on the VLM's answer probability;
3. a prompted verify-or-update controller;
4. **describe-then-judge** (pilot S2), which is the strongest cheap abstention baseline in the pilot;
5. Setting A only: the full-context VLM, as a reference point.

**Metrics:**
- missed updates and false revisions;
- update latency, measured from the annotated evidence span to the moment the belief flips;
- QA accuracy, STORM-BR, the risk–coverage curve and AUROC;
- inference cost, counted as VLM calls and inspections per question;
- results per domain: Cook, Bike, Health, Music and Sports for real video, plus THOR and VHome for simulation.

**Ablations:**
- text-only entries versus text plus visual anchor;
- no drift (h = 0);
- no reliability weight (r = 1);
- the deterministic four-way controller;
- no inspection;
- 3B versus 8B backbone.

## 9. Compute and data risks

- **Data.** The full STORM-Bench video release currently returns 404 on Hugging Face. Only 7 example episodes are available, with 50 real-video questions. Contact the authors now. If the release does not arrive, make OVO-Bench and ProactEgoMem the main real-video evaluation and report STORM on the examples only.
- **Compute.** The pilot ran 8B in 4-bit on an Apple M4 at about 40 s per query with 32 frames. Feature extraction must move to a GPU node and be run once into a cache. The update module is small (under 5M parameters) and trains on cached features.
- **Statistical power.** Pilot intervals are about ±0.15. Treat the pilot as motivation only and rerun the probes on the full evaluation set.

## 10. Plan (weeks from now)

1. Get data (contact STORM authors, set up OVO and ProactEgoMem), set up the GPU, and build the feature-caching script by extending `gonogo/probe.py`.
2. Generate matched simulated episodes with determinability labels, and implement the baselines (reuse `status_probe.py` for describe-then-judge).
3. Train the belief memory and run Setting A.
4. Run Setting B with the K sweep and the eviction policies.
5. Ablations, calibration on the real split, and the write-up.
