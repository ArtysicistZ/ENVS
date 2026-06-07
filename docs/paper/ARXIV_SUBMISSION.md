# ArXiv Submission Guide for ENVS Paper

## Pre-Submission Checklist

- [ ] Paper compiles: `pdflatex ENVS.tex && bibtex ENVS && pdflatex ENVS.tex && pdflatex ENVS.tex`
- [ ] PDF shows author names and affiliations
- [ ] All 3 figures render (pipeline, behavior-gated, data-efficiency)
- [ ] Bibliography complete (no `[?]` markers)
- [ ] Archive created via `bash build_arxiv.sh`

## Submission Package

The archive `ENVS-arxiv.tar.gz` contains 6 files:

| File | Description |
|------|-------------|
| `ENVS.tex` | Main paper (standalone article class) |
| `ENVS.bbl` | Pre-compiled bibliography (recommended — guarantees byte-identical refs) |
| `references.bib` | Bibliography source (fallback; ArXiv auto-runs BibTeX too) |
| `pipeline.png` | Pipeline overview figure |
| `data-efficiency.png` | Data efficiency curve |
| `behavior-gated.pdf` | Behavior-gated branching diagram |

No external style files needed -- uses standard `article` class with common LaTeX packages.

## Account & Endorsement

1. **Register** at [arxiv.org](https://arxiv.org) with institutional email.
2. **Endorsement** (required for first-time submitters):
   - As of Jan 2026: institutional email **AND** prior ArXiv authorship, **OR** personal endorsement from an established ArXiv author.
   - Find endorsers via "Which authors of this paper are endorsers?" on any ArXiv abstract page.

## Category Selection

| Priority | Category | Rationale |
|----------|----------|-----------|
| Primary | `cs.AI` | Agent-centric contribution (GUI agent training) |
| Cross-list | `cs.LG` | RL methodology (search-and-filter, policy optimization) |
| Cross-list | `cs.CV` | Vision/screenshot processing |

## Upload Process

1. Go to [arxiv.org/submit](https://arxiv.org/submit)
2. Upload `ENVS-arxiv.tar.gz`
3. ArXiv auto-detects `ENVS.tex` as toplevel, `pdflatex` as compiler
4. Review auto-compiled PDF
5. Enter metadata:
   - **Title**: ENVS: Environment-Native Verified Search for Long-Horizon GUI Agents
   - **Abstract**: Copy from ENVS.tex (max 1920 chars, ASCII + TeX math)
   - **Authors**: Yincheng Zhou, Athena Zhuoming Zhong, Shijie Zhang, Kevin Zhang, Teresa Xiaotao Shang
   - **License**: CC-BY 4.0 (recommended)
   - **Comments**: 19 pages, 3 figures
6. Preview and submit

## Timing

- Submit before **14:00 US Eastern** to appear at **20:00 Eastern** same day.
- Use **"Unsubmit"** button for pre-announcement corrections.

## ArXiv Environment

| Component | Version |
|-----------|---------|
| System | Submission 1.5 |
| TeX Live | 2025 (frozen 2025-08-03) |
| Compiler | pdflatex |
| BibTeX | Auto-runs from .bib files |
