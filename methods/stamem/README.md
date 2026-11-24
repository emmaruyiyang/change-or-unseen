# StaMem v0: state-aware memory under a frame budget

Status 2026-11-17: the framework runs end to end on the 7 STORM-Bench example episodes. EgoStream is the intended
benchmark, but its question annotations are not released yet (repo `coai-iplab/EgoStream`, `data/annotations` says
"will be released"). Its code, configs and the 9,804-token budget setting are public.

## What is built

| File | Role |
|---|---|
| `stream.py` | Episodes as 1 fps frames; one CLIP ViT-B/32 embedding per frame (cached in `cache/`). This is the cheap per-frame signal: no VLM call per frame |
| `policies.py` | Write/evict policies under a budget of K frames: `fifo` (sliding window), `uniform` (keep frames spread over time), `dedup` (similarity pruning), `stamem` |
| `coverage.py` | No-VLM go/no-go: how much question evidence each policy keeps, and how many kept slots are near-duplicates |
| `answer.py` | The accuracy metric: a frozen VLM answers from only the kept frames (StaMem optionally adds a state tag per frame). `--data ovo/videomme` for long videos |
| `fetch_long.py` | Fetches single long videos by HTTP range request (OVO-Bench `src_videos.tar` parts, Video-MME `videos_chunked_*.zip`) instead of whole archives |
| `longdata.py` | Long-video episodes (OVO backward questions at t ≥ 120 s; Video-MME long), 1 fps frames in one ffmpeg pass, batched CLIP embeddings |

## StaMem v0 (training-free)

Each memory entry is one keyframe of one view, plus a tag: when the state was first seen, the last time it was seen
unchanged, and whether a later frame of the same view showed a different state (then the entry is kept as history).
For each new frame, compare its embedding with the current entries:

- similarity ≥ τ_same (0.93) → **Keep**: nothing changed; extend the entry's "last seen unchanged" time, no new slot.
- τ_view (0.80) ≤ similarity < τ_same → **Update**: same view, new state; write it and mark the old entry as history.
- similarity < τ_view → **Add**: a new view.
- Over budget → evict history before current entries; within a kind, evict the most redundant entry.

The newest frame always stays visible.

## Commands

```bash
.venv/bin/python methods/stamem/stream.py                 # embed frames, print similarity stats
.venv/bin/python methods/stamem/coverage.py --sweep       # no-VLM evidence coverage per policy, K = 2, 4, 8
.venv/bin/python methods/stamem/answer.py --K 4           # VLM answers from kept frames (Qwen2.5-VL-3B, MLX)
.venv/bin/python methods/stamem/answer.py --K 4 --summarize
.venv/bin/python methods/stamem/fetch_long.py ovo --n 20       # long-video pilot data
.venv/bin/python methods/stamem/longdata.py ovo                # frames + CLIP
.venv/bin/python methods/stamem/answer.py --data ovo --K 8     # resumable; add --summarize for the table
```

## First results (2026-11-17, STORM-Bench examples, budget K = 4 frames)

**No-VLM coverage** (real episodes, questions whose prefix has more than K frames; `coverage.py`):

| K | policy | evidence kept (time) | evidence kept (view) | wasted slots |
|---|---|---|---|---|
| 4 | fifo | 0.64 | 0.68 | 0.32 |
| 4 | uniform | **0.77** | **0.84** | 0.23 |
| 4 | dedup | 0.63 | 0.70 | 0.14 |
| 4 | stamem | 0.63 | 0.73 | **0.01** |
| 8 | uniform | **0.91** | **0.96** | 0.25 |
| 8 | stamem | 0.83 | 0.94 | **0.05** |

**VLM accuracy** (Qwen2.5-VL-3B, MLX; `answer.py`; 426 answers):

| system (K = 4) | real, all (50) | real Known (39) | sim (21) |
|---|---|---|---|
| full prefix (≤ 32 frames) | 0.60 | 0.72 | 0.52 |
| fifo | 0.60 | 0.69 | 0.52 |
| uniform | 0.58 | 0.69 | 0.52 |
| dedup | **0.62** | 0.72 | 0.43 |
| stamem | 0.58 | 0.67 | 0.48 |
| stamem + tags | 0.58 | 0.67 | 0.48 |

What this says:

1. **STORM cannot tell memory policies apart.** Four frames already match the full prefix (0.60 vs 0.60). The real
   episodes are 8 to 43 s long, so there is nothing for a memory policy to win. All differences are one or two
   questions. This is the pilot's G2 result again.
2. **StaMem v0 wastes almost no slots but keeps less evidence than uniform.** Fewer near-duplicates did not mean
   more useful frames: uniform keeps 23% duplicates and still keeps more evidence. On these short clips StaMem logs
   on average 8.8 keeps, 5.4 updates and only 1.8 new views per question, so it sees one view whose global CLIP
   embedding drifts with camera motion and occlusion. That embedding cannot tell "not visible" from "changed", and
   it is too coarse for small object-state changes. Evicting history first or adding a temporal-gap term changed
   coverage by at most 0.03.
3. **The tags changed no answers** with the 3B model.
4. The τ sweep on the simulated episodes was flat (their evidence sits next to the query time), so the thresholds
   are not tuned.

Next: a benchmark with long streams (EgoStream once its annotations are out; until then a long-video set such as
OVO-Bench or Video-MME long videos), and a trigger that can tell "not visible" from "changed" (a visibility score,
as in the belief memory, or region-level features instead of one global embedding).

## Long-video pilot (2026-11-17, OVO-Bench, budget K = 8 frames)

Data fetched and prepared (frames + CLIP cached), but only part of it answered, by choice: this is a pilot.

- OVO-Bench: 20 videos (mostly 8-min Ego4D clips), 91 backward questions (EPM/ASI/HLD) asked at t ≥ 120 s.
  **Answered: the first 35 questions, all EPM** ("where is X now"), all 6 systems (211 answers).
- Video-MME long: 20 videos (30–55 min, 4 per domain), 60 questions; frames and CLIP ready, **not answered**.
- Qwen2.5-VL-3B (MLX) takes about 17 s per 8-frame answer and 73 s for the 32-frame reference, so the full
  set would take about 5 hours.

| system (OVO EPM, K = 8, n = 35) | accuracy | oldest kept frame (median) |
|---|---|---|
| full (32 frames spread over the prefix) | 0.46 | – |
| fifo (last 8 s) | **0.69** | t = 303 s |
| uniform | 0.37 | t = 0 s |
| dedup | 0.49 | t = 36 s |
| stamem | 0.60 | t = 157 s |
| stamem + tags | 0.54 | – |

What this says (n = 35, standard error about ±0.08, one question type):

1. **Long streams do separate the policies** (0.37 to 0.69), unlike STORM. The headroom is there.
2. **Recency wins on EPM.** These questions ask where an object is now, so the answer is usually in the last
   seconds. Spreading frames over the whole prefix (uniform, and even the 32-frame reference) dilutes it.
3. **StaMem is second.** It keeps the newest state plus some older views (median oldest frame 157 s), which beats
   uniform and dedup but not FIFO. It has no notion of which past state a question needs.
4. **The tags again did not help** (0.54 vs 0.60).

Open: ASI/HLD (which need older content, where FIFO should lose) and Video-MME are not answered yet. A fair
comparison needs a mix of "current state" and "past event" questions; resume with the commands above.
