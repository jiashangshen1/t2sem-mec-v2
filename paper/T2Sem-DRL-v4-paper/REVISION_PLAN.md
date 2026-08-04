# 修订行动清单（Response to Meta-Review）

**目标叙事：** 从「新 DRL 框架」转向 **「模型知情约束接口下的 per-tier 语义 SLA 保障机制；RL 为可选优化层」**。

**当前稿件状态评估：** Weak Accept / Major Revision  
**修订后目标：** 用诚实贡献层次 + 更强基线 + mobility 必要性论证 → Accept 潜力

---

## 一、必须做（Revision 成败关键）

### M1. 重写贡献声明（诚实量化层次）

| 机制 | 证据 | Overall Δ | 审稿人定位 |
|------|------|-----------|------------|
| **可行域掩码 F** | full 90.4% vs w/o mask 61.2% | **+29.2 pp** | **主杠杆 #1** |
| **语义压缩 μ 模型** | full vs w/o semantic 47.0% | **+43.4 pp** | **主杠杆 #2**（无掩码时仍需语义） |
| **RL（masked PPO）** | T2Sem-DRL 88.3% vs masked greedy 80.5% | **+7.8 pp** | 次要：在 F 之上的策略优化 |
| **per-tier λ** | full 90.4% vs single-λ 89.7% | **+0.7 pp overall** | 边际：主要改善分档均衡（Table II） |
| **双时间尺度 DDPG** | full vs single-timescale 88.1% SLA | **≈0 pp SLA**；能耗 ×10 | **默认场景非必要** → 移扩展 |

**文稿动作：**
- [x] Abstract / Introduction / Contributions 按上表重排（见 `draft_full.md` v4.2 + `export_ieee_word.py`）
- [x] 新增 **Table V：Cumulative component build-up**（主文）
- [x] 正文明确写：**「DRL 边际增益 < 掩码 < 语义」**

### M2. 补强基线

| 基线 | 现状 | 修订动作 | 优先级 |
|------|------|----------|--------|
| **Masked greedy / random** | ✅ 已有 | 保持；主表注明共享 F | — |
| **single-λ CMDP** | 消融有 89.7% | **升入主表 Table I 或 II** | P0 |
| **TJCCT** | 与 greedy 完全相同（实现缺陷） | **重实现** 凸优化轨迹 + 短尺度匹配；或 **主表移除**、脚注说明并放附录 | P0 |
| **Semantic DRL SOTA** | 缺失 | 复现/对接 **MAPPO semantic offload [3]** 或 **VTC'25 air-ground semantic PPO [22]**（同掩码 F） | P0 |
| **Pure DRL** | 消融 57.4% | 保留作「无 F 上界」对照 | — |

**代码/实验 TODO：**
```bash
# 待跑（revision 分支）
python experiments/run_all.py --variant semantic_mappo_masked --config configs/improved_v4_pertier.yaml
python experiments/run_all.py --variant single_lambda_main --eval-only  # 升入主表
python experiments/run_all.py --variant tjcct_full  # 需实现独立轨迹模块
```

### M3. 论证或降级 two-timescale（DDPG）

**数据事实：** single-timescale 88.1% SLA @ **3.3 kJ** vs full 90.4% @ **33 kJ**（当前负载下 near-hover 近最优）。

**二选一（建议 A + 部分 B）：**

- **A（推荐，已在稿中）：** 主文默认 **single-timescale / P4 高能压** 为「推荐运行点」；DDPG 移至 **Section VII-G Extension** 或 Supplement。
- **B（补强论证）：** 设计 **mobility-critical 场景**：
  - 提高 UE 速度 / 热点迁移（waypoint speed ×3）
  - 降低 HAP 可用率或增大区域（UAV 位移扩大可行域）
  - 指标：DDPG vs fixed-hover 的 **interactive-tier SLA** 与 **feasible-set coverage**

**文稿动作：**
- [x] Conclusion / Limitations 写明 DDPG 边际价值未在当前负载下隔离证明
- [ ] 新增 mobility-critical 实验（1 组即可）

### M4. 统一能耗报告口径

**问题：** 主表「88.3% SLA + 33 kJ」vs greedy「80.5% + 4 kJ」口径不一（含轨迹探索能耗）。

**修订方案（三行并列表）：**

| 运行点 | SLA | Energy | 用途 |
|--------|-----|--------|------|
| **R1 服务默认** | 88.3% | 33.5 kJ | 完整训练 checkpoint（探索性能） |
| **R2 能耗调优（P4）** | 91.1% | 16.2 kJ | **主文推荐部署点** |
| **R3 无 DDPG** | 88.1% | 3.3 kJ | 与启发式公平能耗对比 |

**统一指标（新增）：**
- **E_SLA = Energy / N_satisfied**（每满足 SLA 任务能耗，J/task）
- 主表 **禁止** 仅用 R1 与 greedy 比能耗

**文稿动作：**
- [x] Table I 拆分为 Table I（SLA/时延）+ Table I-b（能耗运行点）
- [x] 全文搜索删除「能耗低于/优于 random」类表述（Conclusion/Limitations 已加 non-claim）

---

## 二、建议做（提升接收概率）

### S1. 算法伪代码 + 资源争用模型
- [ ] Algorithm 1：Train T2Sem-Mask-CMDP（F → PPO → λκ ascent）
- [ ] Algorithm 2：Feasibility projection Π_F（逐步）
- [ ] Section III 补充 FIFO 争用公式与 proportional share _floor

### S2. 约束违约率 & λκ 收敛
- [ ] 从训练 log 绘制 **per-tier violation rate** vs episode
- [ ] 绘制 **λ_critical, λ_interactive, λ_best_effort** vs episode
- [ ] 说明有限 horizon 下 CMDP 理论约束的 empirical 近似

### S3. 可复现 + 统计检验
- [ ] 发布 `improved_v4_pertier.yaml` + seeds {42,123,456}
- [ ] 主表 ±std + **Wilcoxon / bootstrap 95% CI**（3 seed 至少报告 CI）
- [ ] `README` 一键复现命令

---

## 三、叙事改写对照（Before → After）

| 维度 | Before（v4.1） | After（v4.2 修订） |
|------|----------------|-------------------|
| 标题重心 | Two-timescale DRL | Model-informed per-tier semantic SLA interface |
| 第一贡献 | 双时间尺度 PPO–DDPG | **可行域 F + 系统级 action interface** |
| 第二贡献 | per-tier CMDP | **语义–时延解析模型 + μ 可行性** |
| 第三贡献 | 掩码 | **masked PPO（+7.8 pp over greedy）** |
| 第四贡献 | 实验全面 | per-tier λ（分档均衡）；DDPG **可选** |
| TJCCT | 与 greedy 同列竞争 | 脚注缺陷 / 重跑后对比 |
| 能耗 | 单点 33 kJ | 三运行点 + E_SLA |

---

## 四、审稿意见逐条回复草稿（Rebuttal 骨架）

**Q1：DRL 边际价值不足？**  
R：同意。消融显示掩码 +29.2 pp、语义 +43.4 pp，而 masked PPO 仅 +7.8 pp over masked greedy。修订稿将 DRL 定位为 F 之上的策略层，非核心创新轴。

**Q2：TJCCT 异常？**  
R：承认当前 TJCCT 短尺度与 greedy 共用、长尺度未启用凸优化模块，导致指标相同。修订将重实现或移附录，不占主表。

**Q3：CMDP 理论与有限 horizon 脱节？**  
R：将明确采用 **empirical per-episode tier rate** 驱动 λκ 上升，并补充违约率/λ 曲线作为有限样本近似。

**Q4：为何需要 DDPG？**  
R：默认负载下不需要（single-ts 88.1% @ 3.3 kJ）。修订补充 mobility-critical 场景或把 DDPG 降为扩展实验。

---

## 五、文件变更索引

| 文件 | 状态 |
|------|------|
| `draft_full.md` | v4.2 叙事 + Table I-b + 贡献重排 |
| `scripts/export_ieee_word.py` | 同步 Abstract / Intro / Table / Conclusion |
| `REVISION_PLAN.md` | 本清单 |
| 实验代码 | 待 M2/M3/S2 分支 |

**下一步执行顺序：** M1 文稿 → M4 表格 → M2 补实验 → M3 mobility → S1–S3
