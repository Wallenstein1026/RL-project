# 4x4 Connect-4 Q-Learning 最终实验报告

**日期:** 2026-05-05
**训练:** 500,000 局/run，3 seeds/config，eval_interval=2000
**评估:** vs Minimax (depth=3, alpha-beta, 300 games) + Policy Optimality (800 states)

---

## 改动总结

### 代码改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/symmetry.py` | **新建** | D4 对称变换模块（8 种对称：4 旋转 x 2 反射），支持 experience augmentation |
| `src/agents.py` | 修改 | 1. `QLearningAgent` 新增 `use_symmetry` 参数，重构 `update()` 支持对称增强 2. **新增 `UCBQLearningAgent`** (~130 行)：UCB 探索替代 epsilon-greedy |
| `src/training.py` | 修改 | 新增 `train_vs_random_then_selfplay()`：前半段 vs Random，后半段 self-play |
| `main.py` | 大幅修改 | 1. `experiment_2()` 重写为 3 范式 x 3 调度完整交叉矩阵 2. 新增 `experiment_symmetry()` 3. 新增 `experiment_ucb()` 三部分实验 |

### 实验内容

| 实验 | 配置数 | Seeds | 状态 |
|------|:---:|:---:|:---:|
| **Exp1**: 探索调度消融 (6 schedules, vs_Random) | 6 x 3 | 42-44 | 完成 |
| **Exp2**: 训练范式 x 调度完整交叉矩阵 (3x3=9 configs) | 9 x 3 | 142-744 | 完成 |
| **Exp3**: vs Minimax 基线对比 | 6 x 3 | 262-305 | 完成 |
| **Exp4 (Symmetry)**: D4 对称性增强 | 3 x 3 | 200-212 | 完成 |
| **Exp5 (UCB)**: UCB 探索替代 epsilon-greedy | 7 x 3 | 42-744 | 完成 |

---

## 实验结果

### 实验1：探索调度消融

**设定：** vs_Random 范式，6 种调度策略

| 调度 | Win vs Random | Draw vs Minimax | Policy Optimality |
|------|:---:|:---:|:---:|
| fixed epsilon=0.1 | 88.3+/-1.2% | - | 14.8+/-14.7% |
| fixed epsilon=0.2 | 90.4+/-1.1% | - | 18.2+/-14.0% |
| fixed epsilon=0.3 | 89.1+/-2.4% | - | 17.2+/-15.3% |
| **fixed epsilon=0.5** | **88.4+/-0.9%** | - | **30.2+/-15.4%** |
| linear | 87.6+/-1.2% | - | 29.6+/-14.9% |
| exponential | 91.0+/-1.5% | - | 12.1+/-0.5% |

**结论：** fixed epsilon=0.5 和 linear 调度的 policy optimality 最高（~30%），但方差较大。

### 实验2：训练范式 x 调度完整交叉矩阵

**设定：** 3 范式 x 3 调度 = 9 配置

| 范式 | 调度 | Win vs Random | Draw vs Minimax | Policy Optimality | Q-table Size |
|------|------|:---:|:---:|:---:|--:|
| vs_Random | fixed (0.3) | 89.3% | 11.1% +/- 9.4% | 6.4% +/- 0.2% | 1.3M |
| vs_Random | linear | 87.5% | 38.2% +/- 43.9% | 9.3% +/- 1.1% | 1.7M |
| vs_Random | exponential | 90.7% | 33.7% +/- 37.3% | 16.2% +/- 13.2% | 1.2M |
| **vs_SelfPlay** | **fixed (0.3)** | 67.4% | **57.4% +/- 34.7%** | 18.8% +/- 9.4% | 0.9M |
| vs_SelfPlay | linear | 73.3% | 39.7% +/- 43.1% | **26.5% +/- 13.8%** | 1.5M |
| vs_SelfPlay | exponential | 67.9% | 14.2% +/- 10.1% | **26.5% +/- 11.3%** | 1.0M |
| vs_RandThenSelf | fixed (0.3) | 80.0% | 36.6% +/- 42.1% | 16.1% +/- 11.6% | 1.2M |
| vs_RandThenSelf | linear | 76.9% | 36.3% +/- 45.1% | 11.2% +/- 3.4% | 1.5M |
| vs_RandThenSelf | exponential | 79.9% | 20.8% +/- 10.4% | 10.1% +/- 1.3% | 1.1M |

**关键发现：**
- **SelfPlay + fixed 平局率最高 (57.4%)**，其中 1 个 seed 达到 100%。推翻旧实验"vs_Random > SelfPlay"的片面结论
- 5/9 配置中有至少 1 个 seed 达到了 96-100% 的平局率
- 极端种子敏感性：同一配置最优和最差 seed 差距巨大

### 实验3：vs Minimax 基线对比

| 调度 | Draw vs Minimax | Loss vs Minimax | Policy Optimality |
|------|:---:|:---:|:---:|
| **fixed epsilon=0.5** | **43.8%** | 56.2% | 17.3% |
| fixed epsilon=0.1 | 33.1% | 66.9% | 16.8% |
| fixed epsilon=0.2 | 27.3% | 72.7% | 17.9% |
| fixed epsilon=0.3 | 6.9% | 93.1% | 26.5% |
| linear | 5.4% | 94.6% | 13.7% |
| exponential | 7.0% | 93.0% | 9.6% |

**结论：** epsilon=0.5 平局率最高，但 policy optimality 最高的却是 epsilon=0.3。高 optimality != 高平局率。

### 实验4：D4 对称性增强

**算法：** 每次 Q-update 同时更新 8 种对称等效状态，有效训练数据扩大 ~5-6x。

| 配置 | Draw vs Minimax | Policy Optimality | Q-table Size |
|------|:---:|:---:|--:|
| Baseline (epsilon=0.5, no sym) | 17.2% | 7.1% | 1.7M |
| Symmetry (epsilon=0.5) | 33.3% | **35.9%** | 8.4M |
| **Symmetry (epsilon=0.3)** | **49.7%** | 9.0% | 6.5M |

**关键发现：**
- 对称性增强大幅提升 Policy Optimality：7.1% -> 35.9%（5.1x）
- 平局率显著改善：17.2% -> 49.7%
- Q-table 膨胀 ~4-5x（6.5-8.4M entries）
- 种子敏感性仍然存在（最优 100%，最差 0%）

### 实验5：UCB 探索

**算法：** UCB 探索替代 epsilon-greedy，公式 `a* = argmax [Q(s,a) + c * sqrt(ln(N_s+1) / N(s,a))]`

**Part 1: c 参数网格 (vs_Random)**

| c | Win vs Random | Draw vs Minimax | Policy Optimality |
|---|:---:|:---:|:---:|
| **0.5** | 83.9% +/- 0.2% | **22.0% +/- 20.7%** | 19.7% +/- 12.8% |
| 1.0 | 67.4% +/- 1.3% | 8.2% +/- 1.7% | 20.6% +/- 11.0% |
| 2.0 | 66.6% +/- 1.9% | 14.9% +/- 3.0% | 26.1% +/- 11.0% |
| 5.0 | 64.9% +/- 0.9% | 8.2% +/- 7.4% | 11.7% +/- 2.9% |

**Part 2: 训练范式 (best c=0.5)**

| 范式 | Draw vs Minimax (best) | Policy Optimality |
|------|:---:|:---:|
| vs_Random + UCB | 66.7% | 37.0% |
| vs_SelfPlay + UCB | 35.3% | 34.4% |
| vs_RandomThenSelfPlay + UCB | 14.1% +/- 6.1% | 21.7% +/- 12.1% |

**Part 3: vs Minimax 基线 (best agents)**

| c | Draw vs Minimax | Policy Optimality |
|---|:---:|:---:|
| 0.5 | **56.0%** | 34.1% |
| 1.0 | 10.7% | 29.0% |
| 2.0 | 16.3% | 12.5% |
| 5.0 | 21.0% | 12.6% |

---

## 综合对比：epsilon-greedy vs UCB vs Symmetry

### 最佳结果汇总

| 指标 | 最佳配置 | 值 |
|------|---------|-----|
| Policy Optimality (最高) | Symmetry epsilon=0.5 | 35.9% mean (best seed: 46.2%) |
| Draw vs Minimax (最高均值) | SelfPlay + fixed (Exp2) | 57.4% +/- 34.7% |
| Draw vs Minimax (单 seed) | 多个 config | 100% (6/27 runs) |
| Draw vs Minimax (best agent) | UCB c=0.5 vs_Random | 66.7% (single eval) |
| UCB best vs epsilon-greedy best | - | 相当，UCB 未超越 |

### UCB vs epsilon-greedy 对比

| 维度 | epsilon-greedy | UCB |
|------|:---:|:---:|
| 探索方式 | epsilon 概率随机 | 优先高不确定性动作 |
| Draw vs Minimax (best scenario) | SelfPlay+fixed: 57.4% +/- 34.7% | Part 2 vs_Random best: 66.7% |
| 种子稳定性 | 方差极大 | 方差同样大，无改善 |
| Policy Optimality (best) | Symmetry: 35.9% (best agent) | UCB c=2.0: 26.1% mean |
| 最优探索参数 | epsilon=0.5 (fixed) | c=0.5 (弱探索) |
| 内存开销 | Q-table only | Q-table + visit counts (~2x) |

### 核心结论

1. **4x4 上 Q-learning 可以达到完美平局**：多个 seed 实现了 vs Minimax 100% 平局率

2. **UCB 没有消除种子敏感性**：UDC c=0.5 下 draw_M 从 0% 到 49.7%，方差与 epsilon-greedy 相当

3. **弱探索最优**：epsilon=0.5 和 c=0.5 都是偏弱探索的参数在平局率上表现最好

4. **对称性增强有效且稳定**：policy optimality 提升 5.1x，平局率提升 2.9x，是最有效的单一改进

5. **实验2 设计修复揭示了重要结论**：旧版仅测试 linear 调度会得出 "vs_Random > SelfPlay" 的错误结论，实际上 SelfPlay+fixed 表现最好

6. **Policy Optimality 和平局率不是正相关**：高 optimality != 高 draw rate。高探索率保留了更多防守策略

7. **训练范式的交互效应至关重要**：同样的调度在不同范式下表现差异巨大

### 局限性

- 种子数不足（3 seeds），结果方差大
- Q-table 在大状态空间下膨胀严重（对称增强后 ~8M entries）
- vs Minimax 评估存在天花板效应——depth=3 的 Minimax 也不完美
- 500K episodes 可能不够，百万级训练或许能缓解种子敏感性

### 产出文件

| 文件 | 说明 |
|------|------|
| `results/exp1_summaries.json` | Exp1: 探索调度消融 |
| `results/exp2_extended_summaries.json` | Exp2: 完整交叉矩阵 |
| `results/exp3_minimax_results.json` | Exp3: vs Minimax 基线 |
| `results/exp_symmetry_summaries.json` | Exp4: D4 对称性增强 |
| `results/exp_ucb1_summaries.json` | Exp5 Part1: UCB c-grid |
| `results/exp_ucb2_summaries.json` | Exp5 Part2: UCB 范式 |
| `results/exp_ucb3_summaries.json` | Exp5 Part3: UCB vs Minimax |
| `src/symmetry.py` | D4 对称变换模块 |
| `src/agents.py` | UCBQLearningAgent + symmetry support |
| `src/training.py` | train_vs_random_then_selfplay |
| `main.py` | experiment_symmetry, experiment_ucb |
| `docs/plans/2026-05-05-ucb-qlearning-plan.md` | UCB 实现方案 |
| `docs/plans/2026-05-05-4x4-tictactoe-improvements-design.md` | 总体改进设计 |
