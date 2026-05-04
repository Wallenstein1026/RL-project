# 4x4 Connect-4 Tic-Tac-Toe Extension

This folder is an independent extension of the `3tictactoe` baseline. The
baseline folder is not modified.

## What changed

- Board size changed from 3x3 to 4x4.
- Action space changed from `0-8` to `0-15`.
- Winning rule changed to connect 4 in a row, column, or diagonal.
- Winning lines are generated dynamically from `BOARD_SIZE` and `WIN_LENGTH`.
- Interactive play now prints a 4x4 position map.
- Results are written to `4tictactoe/results/` even when launched from the repo root.
- The exact 3x3 Minimax baseline was replaced with a depth-limited alpha-beta
  search baseline plus a line-count heuristic.

## Why the search baseline changed

Exact full-width Minimax is cheap on 3x3 Tic-Tac-Toe, but 4x4 has a much larger
state space. Running exact Minimax inside every evaluation would make normal
experiments impractical. The new `MinimaxAgent` keeps the same API, but uses:

- configurable depth-limited search (`depth_limit=3` by default),
- alpha-beta pruning,
- center-first action ordering,
- a heuristic that rewards open 1/2/3/4-in-a-row lines and penalizes opponent
  threats.

As a result, `policy_optimality` in this folder should be interpreted as
agreement with a strong search baseline, not a proof of game-theoretic
optimality.

## Suggested run commands

Quick smoke test:

```powershell
python 4tictactoe/main.py --experiment 1 --episodes 200 --eval-interval 100 --num-runs 1
```

Longer exploratory run:

```powershell
python 4tictactoe/main.py --episodes 50000 --eval-interval 2000 --num-runs 3 --seed 42
```

Interactive play after training:

```powershell
python 4tictactoe/main.py --play --agent-path results/agent_exp2_selfplay.pkl
```

If you run from inside `4tictactoe`, use `python main.py ...` instead.
