# T2Sem-DRL v4 — Paper Package

Complete paper package aligned with the `uav-semantic-mec` draft style. Experimental numbers come from the `t2sem-mec-v2` paper suite.

## Main documents

| File | Description |
|------|-------------|
| **`draft_full.md`** | Full English manuscript (Abstract → Section VIII + appendices + references) |
| `draft.md` | Early short draft (superseded by `draft_full.md`) |
| `outline.md` | Story / roadmap / review risks |
| `references.bib` | BibTeX (25 entries) |

## Figures (PDF + PNG)

| Figure | File stem | Role |
|--------|-----------|------|
| Fig. 1 | `fig0_pipeline` | System / pipeline teaser |
| Fig. 2 | `fig2_main_sla` | Main SLA comparison |
| Fig. 3 | `fig3_per_tier_sla` | Per-tier SLA (core result) |
| Fig. 4 | `fig4_pareto` | Energy–SLA Pareto |
| Fig. 5 | `fig5_mask_sensitivity` | Mask sensitivity |
| Fig. 6 | `fig6_ablation` | Ablation bars |
| Fig. 7 | `fig7_delay_energy` | Delay–energy |

```bash
python scripts/plot_paper_figures.py   # regenerate from data/*.csv
```

## Experiment data (`data/`)

- `experiment_summary.csv` — main comparison
- `per_tier_sla.csv` — per-tier SLA
- `per_tier_sla_baseline_maskoff.csv` — Appendix A
- `pareto_points.csv` — Pareto points
- `mask_sensitivity.csv` — sensitivity
- `ablation_summary.csv` — ablation

## Reproduce experiments

```bash
cd ../..   # t2sem-mec-v2 code root when used from the local workspace
python experiments/run_all.py --config configs/improved_v4_pertier.yaml --variants A1_full --device cpu
```

In the published `t2sem-mec-v2` repository, run from the repo root with the same command.

## Headline numbers (for citation checks)

- Main SLA: **88.3%** (greedy 80.5%)
- Per-tier: critical **87.0%** / interactive **78.4%** / best-effort **100%**
- Ablation −mask: **61.2%** (+29.2 pp from the mask)
- Pareto point: **91.1% SLA @ 16,181 J** (P4 high pressure)

## Overleaf workflow

1. Create an IEEEtran project.
2. Upload `references.bib` and `figures/*.pdf`.
3. Convert `draft_full.md` sections to LaTeX (or `pandoc draft_full.md -o main.tex`).
