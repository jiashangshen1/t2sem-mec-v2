# TD3 Long-Agent Swap — 论文插入材料（paste-ready，2026-10-17）

> **来源与口径**：`t2sem-mec-v2/results/td3_swap/`（FINAL_REPORT.md / summary.csv，2026-09-19 完成，5 种子 × 300 回合 × 30 评估局，v4.3 冻结协议，Windows CPU，驱动 `experiments/td3_swap.py`，提交 5c0a6a3）。
> **插入位置**：VII-H 节，"DDPG extension (optional)" 段落块（Table VI 之后）；Table VII 交叉核对表加一行；references.bib 加 Fujimoto TD3 条目。
> **引用编号**：正文按现有手工编号续用 **[23]**——编译/导出 Word 前请核对编号连续性。
> 已同步插入 `draft_full.md`（本文件留档供审阅/回退；已提交 IEEE 的 docx/PDF 不动）。

---

## A. 正文段落（VII-H，插在 Table VI 段落块之后）

**Long-timescale learner swap (DDPG → TD3).** To further probe learner-dependence, we swap the long-timescale DDPG for TD3 [23] (twin critics, delayed policy updates, target-policy smoothing) under the identical frozen protocol (BC warm-start, per-tier $\lambda$, per-tier mask margins, eval-selected checkpoints; 2 agents × 5 seeds {42,123,456,789,1011} × 300 episodes, 30 evaluation episodes per seed). Overall SLA is statistically indistinguishable (**90.79 ± 2.37%** vs. **90.21 ± 2.09%**) — a stronger off-the-shelf learner does not move the aggregate metric, consistent with the interface-dominance evidence above. TD3 instead shifts the *operating point*: critical **−7.4 pp** / interactive **+9.4 pp** (the critical drop concentrates in one seed, 0.617 vs. 0.78–0.81 for the others), with the lowest mean delay (**0.124 s**) and a **−28%** mean energy reduction (13.9 vs. 19.3 kJ); given the large cross-seed spread (per-seed energy 5.7–38.7 kJ across both agents) and the absence of a bootstrap CI, we report the energy gap as **indicative, not significant**. Practically, TD3 is a legitimate energy-aware deployment alternative for the optional long-timescale mode; scientifically, the flat aggregate SLA reinforces that the remaining headroom lives in the interface and contention handling, not in learner capacity.

**TABLE VIII**
**LONG-AGENT SWAP UNDER THE FROZEN PROTOCOL (2 AGENTS × 5 SEEDS × 300 EPISODES).**

| Long agent | Overall SLA (%) | Delay (s) | Energy (J) | Critical (%) | Interactive (%) | Best-effort (%) |
|-----------|:---------------:|:---------:|:----------:|:------------:|:---------------:|:---------------:|
| DDPG | 90.21 ± 2.09 | 0.138 ± 0.028 | 19,326 ± 11,981 | 83.7 | 86.5 | 100.0 |
| TD3 | **90.79 ± 2.37** | **0.124 ± 0.009** | **13,928 ± 7,532** | 76.4 | **95.9** | 99.97 |

## B. Table VII 交叉核对表新增行

| Long-agent swap | DDPG→TD3, seeds {42,123,456,789,1011}, frozen protocol | 90.79% vs. 90.21% (±2.4/±2.1) | Learner upgrade leaves aggregate SLA unchanged; tier operating point shifts (crit −7.4, int +9.4 pp); energy −28% indicative | support (interface-dominance) |

## C. references.bib 新增条目

```bibtex
@inproceedings{fujimoto2018addressing,
  author    = {Fujimoto, Scott and van Hoof, Herke and Meger, David},
  title     = {Addressing Function Approximation Error in Actor-Critic Methods},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  year      = {2018},
  pages     = {1587--1596}
}
```

## D. 口径备忘（勿写进论文）

- 能耗差 **未做 bootstrap CI**（summary.csv 只有逐种子均值）；若审稿人追问显著性，需回 `results/td3_swap/*/logs/full/seed_*/train_metrics.csv` 或重跑逐回合评估走 `bootstrap_stats.py`。
- critical 下滑主要由 seed 456 贡献（0.6166），逐种子明细在 FINAL_REPORT.md。
- 协议设备为 Windows CPU（与 v4.3 主结果同设备）；4070 GPU 重跑可选，未做。
