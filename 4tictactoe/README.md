# 4x4 Connect-4 Tic-Tac-Toe Q-Learning

This folder contains the full 4x4 extension of the original 3x3 Tic-Tac-Toe
Q-learning project. The goal was to move beyond the nearly solved 3x3 setting
and study how tabular reinforcement learning behaves when the board expands to
4x4 and the winning condition becomes connect 4.

The 4x4 experiments were built incrementally from the 3x3 baseline. This README
records the final setup, the code changes made during development, the commands
used to run the experiments, and the training results available in `results/`.

## Final Setup

- Board: `4 x 4`
- State: tuple of 16 integers
  - `0` means empty
  - `1` means X / player 1
  - `2` means O / player 2
- Action: integer from `0` to `15`, representing one board cell
- Win rule: connect 4 in a row, column, or diagonal
- Reward:
  - win: `+1`
  - loss: `-1`
  - draw: `+0.5`
  - non-terminal move: `0`
- Default training length: `500000` episodes
- Default number of seeds: `3`
- Main evaluation opponent: depth-limited search baseline

Important note: the 4x4 search baseline is not exact full minimax. Full minimax
is much more expensive on 4x4, so the project uses a depth-limited alpha-beta
agent with a heuristic evaluator. In result names, `minimax` or `search`
therefore means this depth-limited search baseline.

## File Structure

```text
4tictactoe/
  main.py                  Experiment entry point and CLI
  README.md                This experiment log
  IMPROVEMENTS.md          Notes from earlier improvement passes
  finish_ucb.py            Helper script used when completing UCB results
  src/
    environment.py         4x4 board, connect-4 win lines, rewards
    agents.py              Q-learning, UCB Q-learning, random, search agents
    training.py            vs-random, self-play, random-then-self-play training
    evaluation.py          Metrics, plots, final summaries
    symmetry.py            D4 board symmetry transformations
  results/
    *.json                 Experiment summaries
    *.png                  Training/result figures
    *.pkl                  Saved best-run agents
```

## How To Run

From the repository root:

```powershell
python 4tictactoe/main.py --experiment selfplay_symmetry --episodes 500000 --eval-interval 2000 --num-runs 3
```

From inside `4tictactoe/`:

```powershell
python main.py --experiment selfplay_symmetry --episodes 500000 --eval-interval 2000 --num-runs 3
```

Quick smoke test:

```powershell
python 4tictactoe/main.py --experiment selfplay_symmetry --episodes 200 --eval-interval 100 --num-runs 1
```

Run all 4x4 experiments:

```powershell
python 4tictactoe/main.py --episodes 500000 --eval-interval 2000 --num-runs 3
```

Play against a saved agent:

```powershell
python 4tictactoe/main.py --play --agent-path results/agent_selfplay_sym_selfplay_symmetry_fixed_0.3.pkl
```

## Metrics

The JSON summaries use these main metrics:

| Metric | Meaning |
|---|---|
| `win_vs_random` | Greedy agent win rate against a random opponent |
| `draw_vs_minimax` | Agent as X / first player draw rate against search |
| `draw_vs_minimax_p2` | Agent as O / second player draw rate against search |
| `policy_optimality` | Fraction of sampled actions matching the search baseline score |
| `q_table_size` | Number of learned `(state, action)` entries |

For earlier experiments, `draw_vs_minimax_p2` is not present because those runs
were completed before the second-player metric was added. The code now records
P2 metrics for newly run experiments.

Saved agents are selected from the best run, not simply the last seed. Selection
prioritizes:

1. `draw_vs_minimax`
2. `draw_vs_minimax_p2`
3. `policy_optimality`
4. `win_vs_random`
5. lower `loss_vs_minimax`

## Development Timeline

### Step 1: Create the 4x4 environment

The original 3x3 baseline was copied into a new `4tictactoe/` folder, then the
environment was generalized to:

- use 16 board cells instead of 9
- generate all length-4 winning lines dynamically
- keep the tabular tuple state representation
- preserve the same training/evaluation API as the 3x3 project

This made the 4x4 code comparable to the 3x3 baseline while increasing the
state-action space substantially.

### Step 2: Rebuild the baseline Q-learning experiments

The first 4x4 experiments reused epsilon-greedy Q-learning against a random
opponent. The fixed epsilon setting was expanded into a grid:

```text
0.1, 0.2, 0.3, 0.5
```

The training length was increased to `500000` episodes and kept at `3` seeds per
configuration to limit runtime while still exposing seed variance.

### Step 3: Save the best run agent

Originally, saved agents could come from the final seed only. This was changed
so each configuration saves the best run according to search draw rate first.
This matters because 4x4 results show high seed variance.

### Step 4: Compare training paradigms

Experiment 2 compared:

- training vs random opponent
- self-play
- random-then-self-play curriculum

Each paradigm was tested with fixed, linear, and exponential epsilon schedules.

### Step 5: Add symmetry augmentation

D4 symmetry augmentation was added in `src/symmetry.py`. Each Q-learning update
can be expanded through rotations and reflections of the board. This increases
sample efficiency but also greatly increases Q-table size.

### Step 6: Add UCB exploration

A `UCBQLearningAgent` was added to compare epsilon-greedy exploration with an
Upper Confidence Bound action-selection rule. UCB was tested over:

```text
c = 0.5, 1.0, 2.0, 5.0
```

The best average UCB setting was `c = 0.5`.

### Step 7: Add SelfPlay + Symmetry and second-player evaluation

The final 4x4 change combined the strongest training paradigm and strongest
state augmentation:

- self-play
- D4 symmetry
- fixed epsilon `0.3` and `0.5`

At the same time, 4x4 evaluation was extended to include the second-player
metric `draw_vs_minimax_p2`, matching the 3x3 result format. The P2 metric is
mainly for structural symmetry with 3x3, not because strong 4x4 second-player
performance was expected.

## Experiment Results

All numbers below are averages over 3 seeds unless otherwise stated.

### Experiment 1: Epsilon Schedule And Fixed Epsilon Grid

File: `results/exp1_summaries.json`

| Config | Win vs Random | Draw vs Search P1 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|
| fixed eps 0.1 | 88.30% | 0.78% | 14.84% | 793754 | 1.33% |
| fixed eps 0.2 | 90.37% | 48.78% | 18.23% | 1068741 | 100.00% |
| fixed eps 0.3 | 89.10% | 10.44% | 17.17% | 1318550 | 29.33% |
| fixed eps 0.5 | 88.43% | 7.22% | 30.24% | 1728464 | 11.67% |
| linear eps | 87.63% | 9.11% | 29.64% | 1645252 | 22.00% |
| exponential eps | 91.03% | 5.89% | 12.10% | 1238958 | 16.33% |

Main finding: `fixed_eps_0.2` had the best average draw rate against the search
baseline, and one seed reached 100% draw. Higher epsilon explored more states
but did not consistently improve search robustness.

### Experiment 2: Training Paradigm Comparison

File: `results/exp2_extended_summaries.json`

| Config | Win vs Random | Draw vs Search P1 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|
| vs Random + fixed | 89.33% | 11.11% | 6.43% | 1338682 | 23.00% |
| vs Random + linear | 87.53% | 38.22% | 9.31% | 1654308 | 100.00% |
| vs Random + exponential | 90.67% | 33.67% | 16.16% | 1249601 | 86.00% |
| SelfPlay + fixed | 67.43% | 57.44% | 18.78% | 911859 | 100.00% |
| SelfPlay + linear | 73.30% | 39.67% | 26.50% | 1493459 | 100.00% |
| SelfPlay + exponential | 67.90% | 14.22% | 26.52% | 979574 | 22.00% |
| RandomThenSelfPlay + fixed | 80.03% | 36.56% | 16.14% | 1204421 | 96.00% |
| RandomThenSelfPlay + linear | 76.87% | 36.33% | 11.17% | 1546765 | 100.00% |
| RandomThenSelfPlay + exponential | 79.87% | 20.78% | 10.08% | 1135181 | 33.00% |

Main finding: `SelfPlay + fixed` produced the best average P1 draw rate
against search at 57.44%. It was weaker against random opponents, but stronger
against the search baseline, which suggests self-play learned more defensive or
strategic behavior.

### Experiment 3: Search-Baseline Agreement

File: `results/exp3_minimax_results.json`

| Config | Draw vs Search P1 | Policy Agreement | Best Seed P1 Draw |
|---|---:|---:|---:|
| vs Random + fixed eps 0.1 | 33.11% | 16.80% | 98.33% |
| vs Random + fixed eps 0.2 | 27.33% | 17.87% | 77.00% |
| vs Random + fixed eps 0.3 | 6.89% | 26.50% | 18.00% |
| vs Random + fixed eps 0.5 | 43.78% | 17.30% | 100.00% |
| vs Random + linear | 5.44% | 13.72% | 10.67% |
| vs Random + exponential | 7.00% | 9.65% | 17.67% |

Main finding: policy agreement and draw rate are not perfectly aligned. Some
agents match the search baseline on sampled local decisions but still perform
poorly in complete games.

### Symmetry-Augmented Q-Learning

File: `results/exp_symmetry_summaries.json`

| Config | Win vs Random | Draw vs Search P1 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|
| baseline fixed 0.5 | 86.37% | 17.22% | 7.13% | 1732631 | 48.00% |
| symmetry fixed 0.5 | 95.53% | 33.33% | 35.86% | 8384431 | 100.00% |
| symmetry fixed 0.3 | 92.60% | 49.67% | 9.00% | 6453729 | 100.00% |

Main finding: symmetry was one of the clearest improvements. It increased win
rate against random and often improved search robustness, but the cost was a
large Q-table.

### UCB Exploration

Files:

- `results/exp_ucb1_summaries.json`
- `results/exp_ucb2_summaries.json`
- `results/exp_ucb3_summaries.json`

UCB c-grid:

| Config | Win vs Random | Draw vs Search P1 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|
| UCB c 0.5 | 83.93% | 22.00% | 19.67% | 2242033 | 49.67% |
| UCB c 1.0 | 67.37% | 8.22% | 20.58% | 2304716 | 10.00% |
| UCB c 2.0 | 66.57% | 14.89% | 26.14% | 2305541 | 19.00% |
| UCB c 5.0 | 64.93% | 8.22% | 11.72% | 2306410 | 18.33% |

UCB training paradigms with best `c = 0.5`:

| Config | Win vs Random | Draw vs Search P1 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|
| vs Random + UCB | 86.03% | 29.44% | 25.39% | 2228370 | 64.67% |
| SelfPlay + UCB | 68.63% | 21.89% | 26.91% | 2312830 | 34.33% |
| RandomThenSelfPlay + UCB | 71.57% | 14.11% | 21.72% | 2305371 | 22.67% |

Best-agent UCB evaluation:

| Config | Draw vs Search P1 | Policy Agreement |
|---|---:|---:|
| UCB c 0.5 | 60.33% | 36.10% |
| UCB c 1.0 | 8.67% | 34.05% |
| UCB c 2.0 | 17.00% | 10.05% |
| UCB c 5.0 | 18.33% | 9.74% |

Main finding: UCB worked best with small exploration pressure. `c = 0.5`
produced a strong best agent, but average performance was less stable than the
best epsilon-greedy self-play settings. Larger `c` values appeared to over-
explore.

### Final Experiment: SelfPlay + Symmetry

File: `results/exp_selfplay_symmetry_summaries.json`

| Config | Win vs Random | Draw vs Search P1 | Draw vs Search P2 | Policy Agreement | Q-table | Best Seed P1 Draw |
|---|---:|---:|---:|---:|---:|---:|
| SelfPlay + Symmetry fixed 0.3 | 77.27% | 33.78% | 4.67% | 17.52% | 4527665 | 100.00% |
| SelfPlay + Symmetry fixed 0.5 | 87.90% | 18.89% | 5.78% | 25.48% | 7990171 | 51.00% |

Per-seed detail:

| Config | Seed | Win vs Random | Draw P1 | Draw P2 | Policy Agreement | Q-table |
|---|---:|---:|---:|---:|---:|---:|
| SelfPlay + Symmetry fixed 0.3 | 300 | 67.30% | 0.00% | 5.33% | 7.29% | 4782878 |
| SelfPlay + Symmetry fixed 0.3 | 301 | 82.00% | 100.00% | 4.00% | 9.60% | 4320572 |
| SelfPlay + Symmetry fixed 0.3 | 302 | 82.50% | 1.33% | 4.67% | 35.69% | 4479546 |
| SelfPlay + Symmetry fixed 0.5 | 310 | 86.50% | 51.00% | 6.00% | 6.28% | 8202490 |
| SelfPlay + Symmetry fixed 0.5 | 311 | 89.60% | 0.00% | 5.67% | 32.10% | 8111740 |
| SelfPlay + Symmetry fixed 0.5 | 312 | 87.60% | 5.67% | 5.67% | 38.08% | 7656284 |

Main finding: combining self-play and symmetry did not outperform the earlier
best average result. `fixed 0.3` produced one excellent seed with 100% P1 draw,
but the other two seeds were weak against search. `fixed 0.5` was more stable
against random and had higher policy agreement, but weaker P1 draw. P2 draw
remained low, around 5%, which is expected because the agent was still mainly
selected and interpreted through the P1 policy.

## Overall Conclusions

The 4x4 environment is much less saturated than 3x3. Strong agents can appear,
but results are highly seed-sensitive.

Best average P1 draw rates against search:

| Rank | Config | Average P1 Draw |
|---:|---|---:|
| 1 | SelfPlay + fixed epsilon | 57.44% |
| 2 | Symmetry fixed 0.3 | 49.67% |
| 3 | fixed epsilon 0.2 | 48.78% |
| 4 | fixed epsilon 0.5 in Exp3 | 43.78% |
| 5 | SelfPlay + Symmetry fixed 0.3 | 33.78% |

Best single saved/best-agent signals:

| Config | Best P1 Draw |
|---|---:|
| fixed epsilon 0.2 | 100.00% |
| SelfPlay + fixed | 100.00% |
| SelfPlay + linear | 100.00% |
| Symmetry fixed 0.5 | 100.00% |
| Symmetry fixed 0.3 | 100.00% |
| SelfPlay + Symmetry fixed 0.3 | 100.00% |
| UCB c 0.5 best-agent evaluation | 60.33% |

Interpretation:

- Training against random opponents gives high random-opponent win rate, but it
  does not reliably produce robust search-baseline performance.
- Self-play improves average search robustness, especially with fixed epsilon.
- Symmetry augmentation is valuable but expensive in memory.
- UCB is sensitive to the exploration constant. `c = 0.5` is best here.
- Policy agreement is useful but should not be treated as the only quality
  metric because it does not always predict full-game draw rate.
- SelfPlay + Symmetry is not the best average result in the current runs, but it
  confirms that the combined setup works and can produce very strong individual
  seeds.
- Second-player 4x4 performance is currently weak. This is acceptable for this
  project phase because the P2 metric was added mainly to make the 4x4 report
  structurally consistent with 3x3.

## Result Artifacts

Key summary files:

```text
results/exp1_summaries.json
results/exp2_extended_summaries.json
results/exp3_minimax_results.json
results/exp_symmetry_summaries.json
results/exp_ucb1_summaries.json
results/exp_ucb2_summaries.json
results/exp_ucb3_summaries.json
results/exp_selfplay_symmetry_summaries.json
```

Key plots:

```text
results/exp1_schedules.png
results/exp1_optimality.png
results/exp2_extended_paradigms.png
results/exp2_extended_optimality.png
results/exp_symmetry_optimality.png
results/exp_ucb1_schedules.png
results/exp_ucb1_optimality.png
results/exp_ucb2_optimality.png
results/exp_ucb3_optimality.png
results/exp_selfplay_symmetry_curves.png
results/exp_selfplay_symmetry_optimality.png
```

Saved agents are stored as `.pkl` files in `results/`. Because these are
generated artifacts, they are intended for local analysis and interactive play
rather than source control.

## Report Notes

For the final report, the most defensible story is:

1. 3x3 was close to saturated, so 4x4 was introduced to create a larger
   exploration space.
2. A direct 4x4 epsilon-greedy baseline could learn to beat random opponents,
   but search-baseline robustness was unstable.
3. More exploration and longer training helped, but seed variance remained high.
4. Self-play improved robustness against the stronger search baseline.
5. Symmetry augmentation improved learning quality but increased memory usage.
6. UCB provided a principled exploration alternative, with `c = 0.5` performing
   best.
7. The final SelfPlay + Symmetry experiment showed that the combined method can
   produce strong individual agents, but did not improve the average result over
   the best earlier configuration.
8. Adding P2 evaluation made the 4x4 result format consistent with 3x3, while
   also showing that second-player robustness remains an open limitation.

