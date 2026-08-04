# T2Sem-DRL v4 — Complete Paper Package

桌面论文完整材料包，格式对齐 `uav-semantic-mec/paper/draft_v2.md`，实验数据来自 `t2sem-mec-v2` 论文套件（2025-07-01 完成）。

## 主文档

| 文件 | 说明 |
|------|------|
| **`draft_full.md`** | **完整英文论文**（Abstract → VIII + 3 Appendices + 25 References + 公式编号） |
| `draft.md` | 早期精简版（已被 `draft_full.md` 取代） |
| `outline.md` | 科研 story / roadmap / 审稿风险 |
| `references.bib` | BibTeX（25 条，可导入 Overleaf） |

## 图表（7 组，PDF + PNG）

| 图 | 文件 | 论文章节 |
|----|------|----------|
| Fig. 1 | `fig0_pipeline` | 系统架构 / pipeline teaser |
| Fig. 2 | `fig2_main_sla` | Table I 可视化 |
| Fig. 3 | `fig3_per_tier_sla` | **核心：分档 SLA** |
| Fig. 4 | `fig4_pareto` | 能耗–SLA Pareto |
| Fig. 5 | `fig5_mask_sensitivity` | 掩码敏感性 |
| Fig. 6 | `fig6_ablation` | 消融柱状图 |
| Fig. 7 | `fig7_delay_energy` | 时延–能耗 |

```bash
python scripts/plot_paper_figures.py   # 从 data/*.csv 重绘全部图
```

## 实验数据 `data/`

- `experiment_summary.csv` — 主对比
- `per_tier_sla.csv` — 分档 SLA
- `per_tier_sla_baseline_maskoff.csv` — Appendix A
- `pareto_points.csv` — Pareto
- `mask_sensitivity.csv` — 敏感性
- `ablation_summary.csv` — 消融

## 复现实验

```bash
cd t2sem-mec-v2
python experiments/run_all.py --config configs/improved_v4_pertier.yaml --variants A1_full --device cpu
```

## 论文核心数字（可直接引用）

- 主对比 SLA：**88.3%**（greedy 80.5%）
- 分档：critical **87.0%** / interactive **78.4%** / best-effort **100%**
- 消融 −mask：**61.2%**（+29.2 pp 来自掩码）
- Pareto 最优：**91.1% SLA @ 16,181 J**（P4 high pressure）

## Overleaf 使用

1. 新建 IEEEtran 项目
2. 上传 `references.bib` + `figures/*.pdf`
3. 将 `draft_full.md` 各节转为 LaTeX（或先用 Pandoc：`pandoc draft_full.md -o main.tex`）

## 写作规范

遵循 `research-paper-writing` skill：一表一信息、claim–evidence 对齐、掩码定位为 system interface。
