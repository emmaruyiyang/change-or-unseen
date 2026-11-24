<div align="center">

<h1>change-or-unseen</h1>

<h3>Memory for streaming video: did the state change, or was it just not seen?</h3>

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![VLM](https://img.shields.io/badge/VLM-Qwen3--VL--8B%20(MLX)-555)
![Status](https://img.shields.io/badge/status-research%20pilot-orange)

</div>

A streaming VLM sees each frame once and has to answer later from what it kept. When an object is out of view, the
memory should not treat "not visible" as "changed". This repo holds two memory methods and the perception probes
that measure how much any memory can gain on a benchmark. CS2883R course project.

## 📰 News

- **2026-11:** MemInStream sample probed: answering from the annotated evidence frames beats seeing the whole history, 0.62 vs 0.48.
- **2026-11:** On OVO-Bench long videos, memory policies finally separate (0.37–0.69); on STORM they do not.
- **2026-11:** STORM go/no-go: perception is good enough with Qwen3-VL-8B, but short clips leave no room for memory.

## 🔬 What the probes show

Each probe gives the same frozen VLM different inputs. If **P2 > P1**, choosing the right frames matters, so memory has something to win.

| Benchmark | Questions | P0 options only | P1 all history | P2 evidence frames | P3 text note |
|---|---:|---:|---:|---:|---:|
| STORM (8–43 s) | 39 | 0.36 | 0.77 | 0.77 | 0.62 |
| OVO-Bench long, EPM | 15 | 0.40 | 0.40 | **0.60** | 0.60 |
| MemInStream | 50 | 0.26 | 0.48 | **0.62** | 0.42 |
| MemInStream, state tracking | 12 | 0.33 | 0.33 | **0.50** | 0.17 |

Long videos leave room for memory, and text notes lose state (P3 on state tracking: 0.17). Samples are small, so read these as a direction, not proof.

## 🧩 What is where

| Folder | What it is |
|---|---|
| [`methods/belief/`](methods/belief) | **Belief memory.** A belief over answers, updated clip by clip and weighted by visibility; 13 learned parameters. [Report](methods/belief/REPORT.md) |
| [`methods/stamem/`](methods/stamem) | **StaMem.** A write and evict policy under a frame budget: Keep / Update / Add by CLIP similarity, history evicted first. [README](methods/stamem/README.md) |
| [`probes/storm/`](probes/storm) | P0–P3 on STORM, plus the shared data loading, prompts and MLX wrapper. [Report](probes/storm/REPORT.md) |
| [`probes/ovo/`](probes/ovo) | P0–P3 on OVO-Bench long videos; `--data meminstream` runs MemInStream |
| [`probes/meminstream/`](probes/meminstream) | Sample download and a self-contained P2 script |
| [`viz/`](viz) | Builds an HTML replay page of clips, questions and model answers |

## 📦 Setup

```bash
git clone https://github.com/emmaruyiyang/change-or-unseen.git
cd change-or-unseen
uv venv && source .venv/bin/activate
uv pip install mlx-vlm open_clip_torch torch numpy pandas pillow certifi
```

Runs on Apple silicon with MLX and needs `ffmpeg`. Videos are not in the repo: the fetch scripts download only the clips each sample needs.

## 🔧 Usage

```bash
# data: MemInStream sample (~0.3 GB) and OVO-Bench long videos (~1 GB)
python probes/meminstream/fetch.py --per-task 3
python methods/stamem/fetch_long.py ovo_extra --n 10

# probes
python probes/meminstream/p2_probe.py --dry-run          # which frames P2 uses, no model
python probes/ovo/ovo_long_probe.py --data meminstream   # P0-P3 on MemInStream

# methods
python methods/stamem/answer.py --data ovo --K 8         # StaMem vs FIFO / uniform / dedup
python methods/belief/fit_eval.py --tag qwen3vl8b        # fit and evaluate belief memory
```

Results land in each folder's `results/`. Same protocol everywhere: 1 fps frames up to the query time, at most 32, long side 448 px, temperature 0.

## 📚 Benchmarks

[STORM-Bench](https://github.com/siruzhong/STORM-Bench) · [OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) · [MemInStream](https://huggingface.co/datasets/zizi32/MemInStream) · [Video-MME](https://huggingface.co/datasets/lmms-eval/Video-MME)
