# 4x4 Tic-Tac-Toe Q-Learning 改进设计文档

**Date:** 2026-05-05
**Deadline:** ~10 hours (results needed today, report due tomorrow)
**Scope:** Line A (complete Exp2 cross-product) + Line B (symmetry-augmented Q-Learning)

---

## 1. Background & Goals

Current results show tabular Q-learning on 4x4 Connect-4 is severely limited:
- Best draw rate vs Minimax: 44% (fixed ε=0.5)
- Policy optimality: < 0.27
- Root cause: state space is too large for 500K episodes to cover

**Two gaps to fix:**
1. **Experimental gap:** Exp2 only tests `exponential` schedule across paradigms. Cannot conclude which paradigm is best without testing `linear` and `fixed` too.
2. **Algorithmic gap:** No state-space reduction. 4x4 board has 8 symmetries (D4 group) that are ignored.

**Goals:**
- [ ] Re-run Exp2 with all 3 schedules × 3 paradigms = 9 configs, 3 seeds each
- [ ] Implement symmetry-augmented Q-learning (8-fold data expansion via D4 symmetries)
- [ ] Run symmetry-enhanced agent and compare vs baseline
- [ ] All results saved as JSON (primary), plots secondary (post-process later)

---

## 2. Line A: Complete Experiment 2 Cross-Product

### 2.1 Current State

| Paradigm \ Schedule | fixed | linear | exponential |
|---|---|---|---|
| vs_Random | ✓ (Exp1) | ✓ (Exp1) | ✓ (Exp1) |
| vs_SelfPlay | ✗ | ✗ | ✓ (Exp2) |
| vs_RandomThenSelfPlay | ✗ | ✗ | ✓ (Exp2) |

### 2.2 Target Matrix (Exp2 Extended)

Run all 9 combinations, 3 seeds each, 500K episodes:

```
paradigms = ["vs_Random", "vs_SelfPlay", "vs_RandomThenSelfPlay"]
schedules = ["fixed", "linear", "exponential"]
# fixed uses DEFAULT_FIXED_EPSILON = 0.30
```

### 2.3 Implementation

Modify `experiment_2()` in `4tictactoe/main.py`:

**Current (simplified):**
```python
for paradigm in ["vs_Random", "vs_SelfPlay", "vs_RandomThenSelfPlay"]:
    agent = QLearningAgent(schedule="exponential", ...)
    # train...
```

**New:**
```python
for paradigm in paradigms:
    for schedule in schedules:
        for run in range(num_runs):
            hp = _hp_for_schedule(schedule, fixed_epsilon)
            agent = QLearningAgent(schedule=schedule, total_episodes=total_episodes, **hp)
            # train with paradigm...
```

Save to: `results/exp2_extended_summaries.json`

### 2.4 Output Schema

```json
{
  "vs_Random + linear_eps": {
    "paradigm": "vs_Random",
    "schedule": "linear",
    "runs": [...],
    "mean": {...},
    "std": {...}
  },
  "vs_SelfPlay + fixed_eps_0.3": {
    ...
  }
}
```

---

## 3. Line B: Symmetry-Augmented Q-Learning

### 3.1 Core Idea

4x4 board has the dihedral group D4 symmetries:
- 4 rotations: 0°, 90°, 180°, 270°
- 2 reflections: horizontal flip, vertical flip
- Total: 8 unique transformations

When agent observes state `s` and takes action `a`, the same logic applies to all 8 symmetric equivalents. By updating all 8 simultaneously, effective training data increases 8×.

### 3.2 Symmetry Definitions

For a 4×4 flat board `b[0..15]` where `b[i] = board[row][col]` with `i = row*4 + col`:

| Transform | Formula |
|---|---|
| Rotate 90° | `new[i] = old[3 - col + row*4]` where `row = i//4, col = i%4` → simpler: `new[i] = old[(3-(i%4))*4 + (i//4)]` |
| Rotate 180° | `new[i] = old[15 - i]` |
| Rotate 270° | `new[i] = old[(i%4)*4 + (3-(i//4))]` |
| Horizontal flip | `new[i] = old[(i//4)*4 + (3-(i%4))]` |
| Vertical flip | `new[i] = old[(3-(i//4))*4 + (i%4)]` |
| Rot90 + Hflip | Compose |
| Rot180 + Hflip | Compose |
| Rot270 + Hflip | Compose |

Action mapping: if action `a` maps to position `(r, c)`, the transformed action is the transformed position flattened.

### 3.3 Implementation Plan

**New file:** `4tictactoe/src/symmetry.py`

```python
from typing import List, Tuple, Callable

TRANSFORMS: List[Callable[[int], int]] = [...]  # 8 board transforms
ACTION_MAPS: List[Callable[[int], int]] = [...]  # corresponding action transforms

def augment_experience(state: tuple, action: int, reward: float,
                       next_state: tuple, done: bool) -> List[Tuple]:
    """Return 8 symmetric variants of one (s,a,r,s',done) tuple."""
    results = []
    for t_idx, (board_tf, action_tf) in enumerate(zip(TRANSFORMS, ACTION_MAPS)):
        s_aug = tuple(board_tf(s) for s in state)  # apply to each cell index? No.
        # Actually: transform the board state itself
        # board is tuple of 16 values; we need to permute positions
        pass
    return results
```

Wait — the board state is a tuple of 16 cell values. To transform it, we permute the positions:

```python
def transform_board(board: tuple, perm: List[int]) -> tuple:
    """Apply position permutation to board state."""
    return tuple(board[perm[i]] for i in range(16))
```

Precompute 8 permutations of `range(16)` for each symmetry.

**Modify `QLearningAgent`:**

Add optional flag:
```python
class QLearningAgent:
    def __init__(..., use_symmetry: bool = False):
        ...
        self.use_symmetry = use_symmetry

    def update(self, state, action, reward, next_state, next_available, done):
        if self.use_symmetry:
            from src.symmetry import augment_experience
            experiences = augment_experience(state, action, reward, next_state, done)
            for s, a, r, ns, d in experiences:
                # call original update logic
                self._update_single(s, a, r, ns, self._get_available(ns) if not d else [], d)
        else:
            self._update_single(state, action, reward, next_state, next_available, done)
```

### 3.4 Training Regime for Symmetry Agent

Best baseline config from Exp3 was `fixed_eps_0.5` (44% draw). We'll compare:

1. **Baseline:** same config, no symmetry
2. **Symmetry-1:** same config, symmetry augmentation ON
3. **Symmetry-2:** symmetry ON + reduced epsilon (0.3) to test if symmetry allows lower exploration

Run 3 seeds each, save to `results/exp_symmetry_summaries.json`.

### 3.5 Expected Impact

- Q-table coverage: ~8× more state-action pairs updated per episode
- Policy optimality: expected 0.25 → 0.50+
- Draw rate vs Minimax: expected 44% → 65%+
- Risk: if implementation bug in symmetry mapping, agent learns nonsense. Must validate with unit tests.

---

## 4. Code Changes Summary

| File | Change |
|---|---|
| `4tictactoe/main.py` | Rewrite `experiment_2()` to iterate schedules × paradigms; add `experiment_symmetry()` |
| `4tictactoe/src/symmetry.py` | **New file**: D4 symmetry permutations, board/action transform, `augment_experience()` |
| `4tictactoe/src/agents.py` | Add `use_symmetry` flag to `QLearningAgent`; refactor `update()` to call `_update_single()` |
| `4tictactoe/src/training.py` | No changes needed (symmetry handled inside agent) |

---

## 5. Data Preservation Plan

**Critical rule: JSON is source of truth. Plots are generated from JSON, never the reverse.**

### 5.1 Existing Results
- `exp1_summaries.json` — keep as-is
- `exp2_summaries.json` — keep as-is (old exponential-only results)
- `exp3_minimax_results.json` — keep as-is

### 5.2 New Results
- `exp2_extended_summaries.json` — 9 configs × 3 runs, full metrics history
- `exp_symmetry_summaries.json` — 3 configs × 3 runs

### 5.3 JSON Schema (all new files)

```json
{
  "config_label": {
    "paradigm": "vs_Random|vs_SelfPlay|vs_RandomThenSelfPlay",
    "schedule": "fixed|linear|exponential",
    "hyperparameters": { "alpha": 0.1, "gamma": 0.9, ... },
    "num_runs": 3,
    "seeds": [42, 43, 44],
    "runs": [
      {
        "run_index": 0,
        "seed": 42,
        "history": [
          {"episode": 2000, "win": 0.5, "draw": 0.3, "loss": 0.2, "optimality": 0.15}
        ],
        "final_metrics": {
          "win_vs_random": 0.85,
          "draw_vs_random": 0.10,
          "loss_vs_random": 0.05,
          "win_vs_minimax": 0.0,
          "draw_vs_minimax": 0.44,
          "loss_vs_minimax": 0.56,
          "policy_optimality": 0.17,
          "q_table_size": 45000
        }
      }
    ],
    "mean": { ... },
    "std": { ... },
    "best_run": { ... }
  }
}
```

---

## 6. Execution Plan & Commands

### Step 1: Implement symmetry.py + agent changes (~30 min)
```bash
cd 4tictactoe
# Edit src/symmetry.py, src/agents.py
```

### Step 2: Unit test symmetry transforms (~15 min)
```bash
python -c "from src.symmetry import *; test_all_symmetries()"
# Must verify: applying any transform 4 times returns identity
```

### Step 3: Extend experiment_2 and run (~4-6 hours)
```bash
# Full run: 9 configs × 3 runs × 500K episodes = 13.5M total episodes
# Estimated time: ~5-6 hours
python main.py --experiment 2 --episodes 500000 --num-runs 3 --seed 42
```

### Step 4: Run symmetry experiment (~2-3 hours)
```bash
# 3 configs × 3 runs × 500K = 4.5M episodes
python main.py --experiment symmetry --episodes 500000 --num-runs 3 --seed 100
```

### Step 5: Validate JSON output (~15 min)
```bash
python -c "import json; d=json.load(open('results/exp2_extended_summaries.json')); print(len(d), 'configs')"
python -c "import json; d=json.load(open('results/exp_symmetry_summaries.json')); print(len(d), 'configs')"
```

### Step 6: Generate plots from JSON (if time permits, else skip)
```bash
python scripts/generate_plots.py  # post-processing script
```

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Symmetry transform bug | Medium | **High** (invalid results) | Unit test: verify 4× rotation = identity |
| Runtime exceeds 10h | Medium | High | Run in parallel: Exp2 and Symmetry on separate terminals if CPU allows |
| JSON corruption during write | Low | Medium | Use atomic write: write to temp file, then `os.rename()` |
| No improvement from symmetry | Medium | Medium | Have fallback: report negative result with analysis |

---

## 8. Fallback Plan

If symmetry does not improve results after implementation:
1. Still report Exp2 extended results (academic value)
2. Analyze why symmetry failed: state coverage histogram, Q-table size comparison
3. Propose future work: MCTS, neural network function approximation

---

**Next step:** Review this design, then proceed to implementation.
