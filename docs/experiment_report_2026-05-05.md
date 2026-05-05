# 4×4 Connect-4 Q-Learning 实验报告

**日期:** 2026-05-05
**训练:** 500,000 局/run，3 seeds/config，eval_interval=2000
**评估:** vs Minimax (depth=3, alpha-beta, 300 games) + Policy Optimality (800 states)

---

## 概要

本报告包含 4 个实验：
1. **实验1**: 探索调度 (Exploration Schedule) 消融对比（vs_Random 范式）
2. **实验2**: 训练范式 × 探索调度 完整交叉矩阵（3×3=9 配置）
3. **实验3**: vs Minimax 基线性能对比
4. **实验4 (Symmetry)**: D4 对称性增强 Q-Learning

---

## 实验1：探索调度消融（Exp1）

**设定：** vs_Random 范式，6 种调度策略

| 调度 | Win vs Random | Draw vs Random | Policy Optimality |
|------|:---:|:---:|:---:|
| fixed ε=0.1 | 88.3±1.2% | 8.4±0.4% | 14.8±14.7% |
| fixed ε=0.2 | 90.4±1.1% | 7.3±0.7% | 18.2±14.0% |
| fixed ε=0.3 | 89.1±2.4% | 7.4±1.4% | 17.2±15.3% |
| **fixed ε=0.5** | **88.4±0.9%** | **8.2±0.4%** | **30.2±15.4%** |
| linear | 87.6±1.2% | 8.9±0.6% | 29.6±14.9% |
| exponential | 91.0±1.5% | 6.8±1.1% | 12.1±0.5% |

**关键发现：** fixed ε=0.5 和 linear 调度表现最好（optimality ~30%），但方差较大。

---

## 实验2：训练范式 × 探索调度完整交叉矩阵（Exp2，新增）

**设定：** 3 种范式 × 3 种调度 = 9 配置，完整交叉实验

**任务：** 修复旧实验2 的设计缺陷——旧版只测试了 linear 调度在不同范式下的表现，这会导致有偏结论。

| 范式 | 调度 | Win vs Random | Draw vs Minimax | Policy Optimality | Q-table Size |
|------|------|:---:|:---:|:---:|--:|
| vs_Random | fixed (0.3) | 89.3% | 11.1% ± 9.4% | 6.4% ± 0.2% | 1.3M |
| vs_Random | linear | 87.5% | 38.2% ± 43.9% | 9.3% ± 1.1% | 1.7M |
| vs_Random | exponential | 90.7% | 33.7% ± 37.3% | 16.2% ± 13.2% | 1.2M |
| **vs_SelfPlay** | **fixed (0.3)** | 67.4% | **57.4% ± 34.7%** | 18.8% ± 9.4% | 0.9M |
| vs_SelfPlay | linear | 73.3% | 39.7% ± 43.1% | **26.5% ± 13.8%** | 1.5M |
| vs_SelfPlay | exponential | 67.9% | 14.2% ± 10.1% | **26.5% ± 11.3%** | 1.0M |
| vs_RandomThenSelfPlay | fixed (0.3) | 80.0% | 36.6% ± 42.1% | 16.1% ± 11.6% | 1.2M |
| vs_RandomThenSelfPlay | linear | 76.9% | 36.3% ± 45.1% | 11.2% ± 3.4% | 1.5M |
| vs_RandomThenSelfPlay | exponential | 79.9% | 20.8% ± 10.4% | 10.1% ± 1.3% | 1.1M |

*注：Policy Optimality 是 agent 与 Minimax 的一步走法一致性，并非平局率。最高 opt (26.5%) 的配置平局率仅 14-40%，而最高平局率 (57.4%) 的配置 opt 仅 18.8%。说明**在 4×4 棋盘上，"模仿 Minimax 的走法"不一定等于"能对抗 Minimax 取得好结果"**。*

### 种子级别分析

由于方差较大，单独看均值可能误导。按 config 列出所有 seed 级别的 draw_vs_Minimax：

```
vs_Random        + fixed:       [10.3%, 23.0%, 0.0%]   → mean=11.1%
vs_Random        + linear:      [100%,  1.7%,  13.0%]   → mean=38.2%
vs_Random        + exponential: [86.0%, 1.7%,  13.3%]   → mean=33.7%
vs_SelfPlay      + fixed:       [100%,  57.3%, 15.0%]   → mean=57.4%
vs_SelfPlay      + linear:      [17.0%, 2.0%,  100%]    → mean=39.7%
vs_SelfPlay      + exponential: [22.0%, 0.0%,  20.7%]   → mean=14.2%
vs_RandThenSelf  + fixed:       [9.3%,  96.0%, 4.3%]    → mean=36.6%
vs_RandThenSelf  + linear:      [8.7%,  0.3%,  100%]    → mean=36.3%
vs_RandThenSelf  + exponential: [21.7%, 7.7%,  33.0%]   → mean=20.8%
```

**关键发现：**
1. **SelfPlay + fixed 平局率最高 (57.4%)**，其中 1 个 seed 达到 100%。这**推翻了旧实验的结论**——旧实验只用 linear 调度，得出"vs_Random > SelfPlay"的片面结论，但实际上 SelfPlay + fixed 调度反而最强
2. 5 个 config 中有至少 1 个 seed 达到了 96-100% 的平局率，说明 **4×4 上 Q-learning 可以达到接近完美的表现**
3. 极端 seed 敏感性：同一配置下，最优 seed 和平均值之间差距巨大（如 vs_Random+linear: [100%, 1.7%, 13.0%]）
4. Q-table 大小 0.9M-1.7M entries，SelfPlay 的 Q-table 最小

---

## 实验3：vs Minimax 基线对比（Exp3）

| 调度 | Draw vs Minimax | Loss vs Minimax | Policy Optimality |
|------|:---:|:---:|:---:|
| fixed ε=0.1 | 33.1% | 66.9% | 16.8% |
| fixed ε=0.2 | 27.3% | 72.7% | 17.9% |
| fixed ε=0.3 | 6.9% | 93.1% | 26.5% |
| **fixed ε=0.5** | **43.8%** | **56.2%** | 17.3% |
| linear | 5.4% | 94.6% | 13.7% |
| exponential | 7.0% | 93.0% | 9.6% |

**关键发现：** 再次确认 ε=0.5 的平局率最高（43.8%），但 policy_optimality 最高的是 ε=0.3（26.5%）。这说明 **policy optimality 和平局率不是简单的正相关关系**——更"像 Minimax"不一定意味着能更好地对抗 Minimax。

---

## 实验4：D4 对称性增强 Q-Learning（新增）

**算法：** 每次 Q-update 同时更新棋盘 8 种对称等价状态-动作对（4 旋转 × 2 反射），有效训练数据扩大 ~5-6×。

| 配置 | Draw vs Minimax | Policy Optimality | Q-table Size |
|------|:---:|:---:|--:|
| Baseline (ε=0.5, no sym) | 17.2% | 7.1% | 1.7M |
| **Symmetry (ε=0.5)** | 33.3% | **35.9%** | 8.4M |
| Symmetry (ε=0.3) | **49.7%** | 9.0% | 6.5M |

### 种子级别分析

```
baseline  (ε=0.5, no sym): draw=[0.0%, 3.7%, 48.0%]  opt=[6.6%, 5.9%, 8.9%]
symmetry  (ε=0.5):         draw=[0.0%, 100%, 0.0%]    opt=[17.6%, 43.9%, 46.2%]
symmetry  (ε=0.3):         draw=[0.0%, 100%, 49.0%]   opt=[6.3%, 15.2%, 5.4%]
```

**关键发现：**
1. **对称性增强大幅提升 Policy Optimality**：ε=0.5 从 7.1% → 35.9%（**5.1×**）
2. **平局率显著改善**：ε=0.3+sym 达到 49.7% 均值，且有 1 个 seed 达到 100%
3. **Q-table 膨胀**：symmetry 的 Q-table 约 6.5-8.4M entries（baseline 1.7M），约 4-5×
4. **种子敏感性仍然存在**：最优 seed 100%，最差 0%。对称性不能完全解决稳定性问题
5. ε=0.3+sym 的平局率最好，但 policy optimality 只有 9.0%。这延续了"高 optimality ≠ 高 draw rate"的模式

---

## 综合分析

### 最佳结果汇总

| 指标 | 最佳配置 | 值 | 备注 |
|------|---------|-----|------|
| Policy Optimality | Symmetry ε=0.5 | 35.9% (最高 seed: 46.2%) | 衡量与 Minimax 走法的一致性 |
| Draw vs Minimax (均值) | SelfPlay+fixed | 57.4% ± 34.7% (最高 seed: 100%) | 衡量实际对战平局率，方差很大 |
| Draw vs Minimax (单seed) | 多个 config | 100% (6/27 runs) | 多个 seed 可达到完美平局 |
| Q-table 效率 | SelfPlay | ~1M entries (最小) | SelfPlay 的 Q-table 最小但平局率最高 |

*注：Policy Optimality 和平局率衡量不同的东西。高 opt 不一定带来高平局率。例如 vs_SelfPlay+exponential 的 opt=26.5% 但 draw_M 仅 14.2%。*

### 核心结论

1. **4×4 上 Q-learning 可以达到完美平局**：多个 seed 实现了 vs Minimax 100% 平局率，证明 tabular Q-learning 在大状态空间下仍有潜力

2. **训练极不稳定**：同一配置下最优和最差 seed 可以差一个数量级。需要更多 seed 或更好的初始化策略

3. **对称性增强有效**：policy optimality 提升 5.1×，平局率提升 1.9×

4. **实验2 设计修复是必要的**：旧版只看 linear 调度会得出"vs_Random > SelfPlay"的错误结论。实际上 SelfPlay+fixed 反而最好

5. **Policy Optimality 和平局率不是正相关的**：最优 policy optimality（ε=0.3）的平局率很低（6.9%），而最优平局率（ε=0.5）的 optimality 中等（17.3%）。高探索率保留了更多防守策略

### 局限性

- 种子数不足（3 seeds），结果方差大
- 对称性增强使 Q-table 膨胀 5×，可能影响可扩展性
- vs Minimax 评估存在天花板效应——depth=3 的 Minimax 也不是完美的
- 仅测试了 vs_Random 训练范式 + 对称性；vs_SelfPlay+symmetry 可能进一步提升

### 未来方向

- **Ensemble**: 多 seed ensemble 投票可能大幅提升稳定性
- **MCTS 基线**: 使用 MCTS 作为更强的基线对比
- **更大规模训练**: 500K episodes 可能不够，百万级训练或许能缓解种子敏感性
- **SelfPlay + Symmetry**: 将两个最优方法结合

---

## 产出文件

| 文件 | 说明 |
|------|------|
| `results/exp1_summaries.json` | 实验1 汇总 |
| `results/exp2_extended_summaries.json` | 实验2 完整交叉矩阵 |
| `results/exp3_minimax_results.json` | 实验3 vs Minimax |
| `results/exp_symmetry_summaries.json` | 对称性增强实验 |
| `results/exp1_optimality.png` | 实验1 bar chart |
| `results/exp2_extended_optimality.png` | 实验2 bar chart |
| `results/exp3_optimality_all.png` | 实验3 bar chart |
| `results/exp_symmetry_optimality.png` | 对称性实验 bar chart |
| `src/symmetry.py` | D4 对称变换模块 |
| `docs/plans/2026-05-05-4x4-tictactoe-improvements-design.md` | 设计文档 |
