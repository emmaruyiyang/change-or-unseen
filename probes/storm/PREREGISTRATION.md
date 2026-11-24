# Perception go/no-go — pre-registration

Written 2026-11-04, before any model output was seen.

## Question
Can a frozen VLM read the object facts our controller needs from real egocentric frames?
If it cannot, a learned belief controller has nothing reliable to work with, and the project
should change its backbone or its scope before building the simulation pipeline.

## Data (pilot)
The full STORM-Bench video release is not reachable (the Hugging Face dataset returns 404, and
GitHub ships annotations only). The pilot uses the 7 example episodes in
`STORM-Bench/data/examples`: 71 questions, of which 50 are real video (Cook, Bike, Health, Music,
Sports; 39 Known, 11 Uncertain) and 21 are simulation (THOR, VHome).
At n = 39, a 95% interval on accuracy is roughly ±0.15. **The pilot can show a clear failure or a
clear pass. Anything near a threshold is reported as inconclusive and rerun on the full release.**

## Probes (same model, same question, frames at 1 FPS, temperature 0)
| id | frames shown | options shown | measures |
|----|--------------|---------------|----------|
| P0 blind | none | yes | language prior |
| P1 online | every second from 0 to query time (STORM protocol, ≤ 32 frames) | yes | current end-to-end ability |
| P2 oracle evidence | only the seconds inside the annotated evidence spans | yes | perception when retrieval is solved |
| P3 extract→answer | evidence frames + question, **no options** → free-text fact; then text-only MCQ from that fact | yes (stage 2 only) | how much is lost when memory is text |

Options are shuffled per question with a fixed, id-derived permutation. Each frame is labelled with
its timestamp in the prompt.

## Backbones
Qwen2.5-VL-3B-Instruct (the proposal's choice) and Qwen3-VL-8B-Instruct (4-bit), both via MLX on an
Apple M4.

## Decision rules (primary set: real-video Known questions, n = 39)
- **G1 perception adequate:** P2 ≥ 0.70 and P2 − P0 ≥ 0.20.
- **G2 memory headroom:** P2 − P1 ≥ 0.10. (If P1 ≈ P2 on Known questions, the controller's value
  must come from Uncertain questions, not from recovering facts.)
- **G3 text memory sufficient:** P3 ≥ P2 − 0.10. Otherwise, store visual evidence in memory.

Verdict for each backbone:
- **GO**: G1 holds.
- **GO with backbone switch**: G1 fails for 3B and holds for 8B.
- **NO-GO**: G1 fails for both; perception, not belief revision, is the bottleneck.
- **Inconclusive**: the bootstrap 95% interval of any G1 quantity straddles its threshold.

Secondary (reported, not used for the verdict): the same numbers on the in-scope types only
(current_state, object_tracking, state_change; n = 22); on Uncertain questions, how often P1
picks the abstention option; and how often P1 abstains on Known questions.
