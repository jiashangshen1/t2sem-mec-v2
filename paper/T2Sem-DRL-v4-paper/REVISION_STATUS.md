# v4.3 修订进度（自动更新）

**预计完成时间（wall clock）**

| 阶段 | 内容 | 预计耗时 | 状态 |
|------|------|----------|------|
| **A** | 代码改动（A1/A2/E3/E2/E4/E5） | ~3 h | ✅ 完成 |
| **B** | 冒烟 + 基线重评（TJCCT 修复验证） | ~1 h | ✅ 完成 |
| **C** | 全量训练 `revision_run.py`（300ep×3seed×6 jobs） | 完成 | ✅ 完成 |
| **D** | 文稿/图表/Word v4.3 同步 | ~2 h | ✅ 完成 |

**你可拿到的新一版：**
- **现在**：代码 + `REVISION_STATUS.md` + 修订版 `draft_full.md` v4.3 框架
- **~B 完成后（数小时）**：更新基线表（TJCCT ≠ greedy）+ Fig.8 λ 曲线
- **~C 完成后（1–2 天）**：完整 `revision_summary.csv` → 主表/消融刷新 → 最终 Word

---

## 已完成代码改动（`t2sem-mec-v2`）

1. **A1** `mask_delay_margin_by_tier`（critical 0.75 / interactive 0.65 / best_effort 0.85）
2. **A2** EMA λ 更新 + deficit clipping（`lambda_ema`, `lambda_deficit_clip`）
3. **E3** Lite-TJCCT：K-means 长尺度 + 优先级 water-filling 短尺度（已与 greedy 区分）
4. **E2** `independent_ppo` 基线（`configs/improved_v4_independent_ppo.yaml`）
5. **E4** `mobility_critical.yaml` + 热点漂移 + HAP 可用率 0.7
6. **E5** `train_metrics.csv` 日志 + `scripts/plot_lambda_curves.py` → Fig.8
7. **E6** `scripts/revision_run.py` 一键修订实验套件

## 运行环境（v4.3 起统一 WSL Ubuntu + CPU）

**平台：** WSL2 Ubuntu  
**Python：** `/root/venvs/ml/bin/python`（torch 2.12+cu130，实验用 `--device cpu`）  
**Windows 任务已停止**，全套实验在 WSL 后台重跑。

```bash
# 一键启动（推荐）
wsl bash "/mnt/d/research/uav dql/code/t2sem-mec-v2/scripts/run_revision_wsl.sh"

# 查看进度
wsl tail -f "/mnt/d/research/uav dql/code/t2sem-mec-v2/results/revision_v43/run.log"
```

预计全套：**~9 小时**（WSL CPU，6 jobs × 3 seeds × 300 ep）

## 运行命令（旧）

```powershell
cd C:\Users\12293\Desktop\cpscources\cps3962\project\t2sem-mec-v2

# 全量修订实验（后台已启动）
python scripts/revision_run.py --device cpu

# 冒烟
python scripts/revision_run.py --smoke

# λ 曲线
python scripts/plot_lambda_curves.py --log results/logs_improved_pertier/full/seed_42/train_metrics.csv --out results/revision_v43/fig8_lambda.png
```

## 初步验证（3 seeds × 30 eval ep，per-tier mask v4.3）

| 方法 | Overall SLA | vs greedy |
|------|-------------|-----------|
| **Greedy** | **87.77%** | — |
| **TJCCT (Lite)** | **86.64%** | **−1.13 pp**（已区分，非重复实现） |
| Random | 78.51% | — |
| Local-only | 86.05% | — |

数据文件：`data/baseline_v43.csv`（与 `t2sem-mec-v2/results/revision_v43/baseline_v43.csv` 同步）
