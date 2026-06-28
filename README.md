# Stanford CS336 — Language Modeling from Scratch · Notes

Comprehensive, illustrated lecture notes for **[Stanford CS336 (Spring 2025)](https://stanford-cs336.github.io/spring2025/)** — *"Language Modeling from Scratch"*, one of the best courses for understanding how modern LLMs are actually built, end to end.

### 📖 Read online → **https://qihongruan.github.io/cs336/**

The full playlist is **17 lectures / ~22 hours of video**. These notes condense it into something you can skim in an afternoon and keep as a reference.

If this is useful, a ⭐ is appreciated!

## What's inside

For every one of the 17 lectures:
- **Comprehensive written notes** — covering tokenization, Transformer architectures, Mixture-of-Experts, GPUs & Triton kernels, parallelism, scaling laws, inference, evaluation, data curation, and alignment (SFT / RLHF / RL).
- **~200 real slide screenshots** pulled at the key moments of each lecture, with captions.
- **Rendered math** (LaTeX/MathJax), **syntax-highlighted code**, section timestamps, and a link back to the source video.

## Lectures

| # | Topic | Notes | Video |
|---|-------|-------|-------|
| 01 | Overview and Tokenization | [Read](https://qihongruan.github.io/cs336/lec01.html) · [.md](notes/lec01.md) | [▶](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) |
| 02 | PyTorch, Resource Accounting | [Read](https://qihongruan.github.io/cs336/lec02.html) · [.md](notes/lec02.md) | [▶](https://www.youtube.com/watch?v=msHyYioAyNE) |
| 03 | Architectures, Hyperparameters | [Read](https://qihongruan.github.io/cs336/lec03.html) · [.md](notes/lec03.md) | [▶](https://www.youtube.com/watch?v=ptFiH_bHnJw) |
| 04 | Mixture of Experts | [Read](https://qihongruan.github.io/cs336/lec04.html) · [.md](notes/lec04.md) | [▶](https://www.youtube.com/watch?v=LPv1KfUXLCo) |
| 05 | GPUs | [Read](https://qihongruan.github.io/cs336/lec05.html) · [.md](notes/lec05.md) | [▶](https://www.youtube.com/watch?v=6OBtO9niT00) |
| 06 | Kernels, Triton | [Read](https://qihongruan.github.io/cs336/lec06.html) · [.md](notes/lec06.md) | [▶](https://www.youtube.com/watch?v=E8Mju53VB00) |
| 07 | Parallelism 1 | [Read](https://qihongruan.github.io/cs336/lec07.html) · [.md](notes/lec07.md) | [▶](https://www.youtube.com/watch?v=l1RJcDjzK8M) |
| 08 | Parallelism 2 | [Read](https://qihongruan.github.io/cs336/lec08.html) · [.md](notes/lec08.md) | [▶](https://www.youtube.com/watch?v=LHpr5ytssLo) |
| 09 | Scaling Laws 1 | [Read](https://qihongruan.github.io/cs336/lec09.html) · [.md](notes/lec09.md) | [▶](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) |
| 10 | Inference | [Read](https://qihongruan.github.io/cs336/lec10.html) · [.md](notes/lec10.md) | [▶](https://www.youtube.com/watch?v=fcgPYo3OtV0) |
| 11 | Scaling Laws 2 | [Read](https://qihongruan.github.io/cs336/lec11.html) · [.md](notes/lec11.md) | [▶](https://www.youtube.com/watch?v=OSYuUqGBQxw) |
| 12 | Evaluation | [Read](https://qihongruan.github.io/cs336/lec12.html) · [.md](notes/lec12.md) | [▶](https://www.youtube.com/watch?v=x-R5l2HsXqM) |
| 13 | Data 1 | [Read](https://qihongruan.github.io/cs336/lec13.html) · [.md](notes/lec13.md) | [▶](https://www.youtube.com/watch?v=WePxmeXU1xg) |
| 14 | Data 2 | [Read](https://qihongruan.github.io/cs336/lec14.html) · [.md](notes/lec14.md) | [▶](https://www.youtube.com/watch?v=9Cd0THLS1t0) |
| 15 | Alignment — SFT/RLHF | [Read](https://qihongruan.github.io/cs336/lec15.html) · [.md](notes/lec15.md) | [▶](https://www.youtube.com/watch?v=Dfu7vC9jo4w) |
| 16 | Alignment — RL 1 | [Read](https://qihongruan.github.io/cs336/lec16.html) · [.md](notes/lec16.md) | [▶](https://www.youtube.com/watch?v=46f2QTDB08Q) |
| 17 | Alignment — RL 2 | [Read](https://qihongruan.github.io/cs336/lec17.html) · [.md](notes/lec17.md) | [▶](https://www.youtube.com/watch?v=JdGFdViaOJk) |

## How these notes are made

Each lecture's **video and transcript** are processed automatically — reading the slides, whiteboard math, and on-screen code — and compiled into this static site. Notes are auto-generated and may contain errors: **verify critical details against the original lectures.**

## Repo layout

- `notes/` — the source Markdown for each lecture (readable directly on GitHub)
- `index.html`, `lec01.html` … `lec17.html` — the compiled static site
- `figs/lecNN/` — slide screenshots used in each lecture page

## Disclaimer & credits

Unofficial study notes. All course material, lectures, and slides are © **Stanford University** and the CS336 teaching staff (Percy Liang, Tatsunori Hashimoto, et al.). Slide screenshots are included for educational reference. This repository is not affiliated with or endorsed by Stanford.

---

Built by [Qihong Ruan](https://qihongruan.github.io/). Feedback and connections welcome — open an issue or reach out.
