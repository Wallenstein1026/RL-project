# UCB Q-Learning 实现方案

## 动机

Exp1/Exp2 的核心问题：epsilon-greedy + Q-learning 训练极不稳定。
同一配置 3 seeds 的 draw_vs_Minimax 可以从 0% 到 100%。
根源在于 epsilon-greedy 的探索是无方向的——10% 概率随机选，可能反复尝试已知的烂动作而错过关键的好动作。

**UCB (Upper Confidence Bound)** 替代方案：
- 探索时有方向性——优先选"访问次数少 OR Q 值高"的动作
- 对每个动作的不确定性有一个量化的"乐观估计"
- 理论上比 epsilon-greedy 收敛更快、更稳定

## 改什么

### 1. `agents.py` — 新增 `UCBQLearningAgent`

不修改现有 `QLearningAgent`，而是新建一个类，避免影响已有实验。

```python
class UCBQLearningAgent:
    """
    Tabular Q-learning with UCB exploration instead of epsilon-greedy.

    Action selection:
        a* = argmax [ Q(s,a) + c * sqrt(ln(N_s) / N(s,a)) ]

    where N_s = total visits to state s, N(s,a) = visits to (s,a).
    If N(s,a) == 0, UCB bonus = +inf → always try unseen actions first.
    """

    def __init__(self, player=1, alpha=0.1, gamma=0.9,
                 c=2.0, use_symmetry=False):
        self.q_table = {}         # (state, action) → Q-value
        self.sa_counts = {}       # (state, action) → visit count
        self.s_counts = {}        # state → total visits
        self.c = c                # exploration parameter
        ...

    def select_action(self, state, available_actions, greedy=False):
        if greedy:
            # Pure exploitation (for evaluation)
            return max(available_actions, key=lambda a: self.get_q(state, a))

        best_action = None
        best_value = -float('inf')

        for a in available_actions:
            q = self.get_q(state, a)
            n_sa = self.sa_counts.get((state, a), 0)
            n_s = self.s_counts.get(state, 0)

            if n_sa == 0:
                bonus = float('inf')
            else:
                bonus = self.c * math.sqrt(math.log(n_s + 1) / n_sa)

            ucb_value = q + bonus

            if ucb_value > best_value:
                best_value = ucb_value
                best_action = a

        return best_action

    def update(self, state, action, reward, next_state, next_available, done):
        # 1. Standard Q-update (same as QLearningAgent)
        # 2. Update visit counts
        self.sa_counts[(state, action)] = self.sa_counts.get((state, action), 0) + 1
        self.s_counts[state] = self.s_counts.get(state, 0) + 1

        # 3. If use_symmetry → augment (same as before)
        ...
```

**关键参数 `c`：**
- c=0.5：弱探索，接近 greedy
- c=2.0（推荐默认）：中等探索
- c=5.0：强探索

**与 epsilon-greedy 的区别：**
| | epsilon-greedy | UCB |
|---|---|---|
| 探索方式 | ε 概率随机 | 优先高不确定性动作 |
| 收敛性 | 随机探索可能重复烂动作 | 系统性地减少不确定性 |
| 超参数 | ε_start, ε_end, schedule | 只需 c（无衰减） |
| 内存开销 | Q-table only | Q-table + 访问计数 |

### 2. `training.py` — 无需修改

UCB agent 实现 `select_action` 和 `update` 接口与现有 `QLearningAgent` 完全一致，
`train_vs_random` / `train_selfplay` 直接兼容。

### 3. `main.py` — 新增 `experiment_ucb`

一个精简的新实验，直接对比 UCB vs epsilon-greedy：

```python
def experiment_ucb(total_episodes, eval_interval, num_runs=3, base_seed=200):
    """
    对比 UCB 探索 vs epsilon-greedy，使用与现有实验一致的 seeds。

    3 组对比，每组用 Exp1/Exp2 的 best config 对应的 seeds：
      A. vs_Random + fixed_eps_0.5  (Exp1 best, seeds=[42,43,44])
      B. vs_Random + UCB c=2.0     (same seeds)
      C. SelfPlay + UCB c=2.0       (Exp2 best paradigm, seeds=[442,443,444])
    """
    configs = [
        ("baseline_vsRandom_fixed0.5",  "eps_greedy", "vs_Random",  42,  fixed_eps=0.5),
        ("ucb_vsRandom_c2.0",          "ucb",        "vs_Random",  42,  c=2.0),
        ("ucb_SelfPlay_c2.0",          "ucb",        "vs_SelfPlay", 442, c=2.0),
    ]
    ...
```

**设计要点：**
- 复用 Exp1/Exp2 的 seeds，**不重跑旧实验**
- 3 个配置 × 3 seeds = 9 runs，约 2 小时
- 结果存 `results/exp_ucb_summaries.json`
- 直接对比 seed 级别：seed 42 在 eps-greedy 下 opt=X%，在 UCB 下 opt=Y%

## 改动的文件

| 文件 | 改动 |
|------|------|
| `src/agents.py` | 新增 `UCBQLearningAgent` 类（~60 行） |
| `main.py` | 新增 `experiment_ucb()` 函数 + CLI `--experiment ucb`（~40 行） |
| `training.py` | **无改动**（接口兼容） |
| `evaluation.py` | **无改动** |

## 预计效果

| 指标 | epsilon-greedy (现状) | UCB (预期) |
|------|----------------------|-------------|
| seed 间方差 | opt ±15%, draw ±40% | 显著缩小 |
| 3-seed draw vs Minimax | 均值不可靠 | 均值更可靠 |
| 训练时间 | 基准 | +10-20%（额外计数开销） |

## 风险

- UCB 的两个 dict（sa_counts, s_counts）增加内存 ~2x Q-table 大小
- c 参数需要调参。如果 c 太大 → 探索过多，收敛慢；c 太小 → 退化为 greedy
- 首次尝试 unseen action 的 bonus=+inf 可能导致 agent 在早期过度探索每个动作
