# T2Sem-DRL v4 — Research Roadmap & Story

> 方法论参考：[pengsida/learning_research](https://github.com/pengsida/learning_research)

## 1. 科研问题（Roadmap 对齐）

**长远目标：** 在层次化 UAV–HAP 语义 MEC 中，实现**多档 SLA 可证明保障**下的**能效卸载**。

**本篇定位（v4）：** 在 v2 双时间尺度 DRL 基础上，解决「单指标 SLA 优化导致分档偏科」和「连续动作空间路由不可行」两个算法瓶颈。

**下一篇可递进：** 多 UAV 协同 MAPPO、用户 mobility 增强场景下「飞行必要性」、联邦语义模型缓存。

## 2. 贡献 Story（先对齐再写段落）

1. **问题重要** — 多终端、多档 SLA、语义压缩、层次卸载同时存在
2. **现有不足** — 单 λ CMDP、无可行域、aggregate SLA 掩盖偏科
3. **方法** — per-tier CMDP + 模型引导可行域 + 双时间尺度 DRL
4. **证据** — 主表 88.3%、分档图、Pareto、10 项消融、掩码敏感性

## 3. Project Checklist

- [x] 确定问题与 baseline 协议（masked 共享可行域）
- [x] 实现 per-tier λ + 可行域掩码
- [x] 主实验 300ep × 3 seed
- [x] 分档 / Pareto / 消融 / 敏感性
- [x] 论文草稿 v4 + 图表
- [ ] Introduction 段落级润色（research-paper-writing）
- [ ] Related Work 补 2025–2026 文献
- [ ] 统计检验（可选：bootstrap CI）
- [ ] Demo / Slides / Rebuttal 预备

## 4. 高风险审稿点 & 应对

| 风险 | 应对 |
|------|------|
| 掩码是否算作弊 | 定位为系统级 action interface；Appendix A 给 mask-off |
| DRL vs masked-greedy 增益小 | 强调分档均衡 + 低时延 + Pareto 支配 |
| 能耗高于 baseline | Pareto + single-timescale 3.3k J；不说 beat random on energy |
| no_contention SLA=1 | 脚注为上界，论证竞争建模必要 |

## 5. 文件索引

| 文件 | 内容 |
|------|------|
| `draft.md` | 英文全文草稿 |
| `data/*.csv` | 实验原始汇总 |
| `figures/*.pdf` | 论文图 |
| `scripts/plot_paper_figures.py` | 重绘图表 |
