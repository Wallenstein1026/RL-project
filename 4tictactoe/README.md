# 4x4 Connect-4 Tic-Tac-Toe Q-Learning

This directory extends the original `3tictactoe` baseline to a 4x4 board where
the winning rule is four connected marks in any row, column, or diagonal.

## Structure

- `main.py` - experiment entry point and interactive play.
- `src/environment.py` - 4x4 board, dynamic win-line generation, rewards.
- `src/agents.py` - Q-learning, random, and depth-limited search agents.
- `src/training.py` - training against random opponents and self-play.
- `src/evaluation.py` - evaluation metrics and plotting utilities.
- `results/` - generated agents, JSON summaries, and figures.

## Run

From the repository root:

```powershell
python 4tictactoe/main.py --eval-interval 2000 --num-runs 3 --seed 42
```

From inside this folder:

```powershell
python main.py --eval-interval 2000 --num-runs 3 --seed 42
```

Quick test:

```powershell
python 4tictactoe/main.py --experiment 1 --episodes 200 --eval-interval 100 --num-runs 1
```

The default training length is now `500000` episodes. Experiments 1 and 3 test
the fixed exploration grid `0.1, 0.2, 0.3, 0.5` by default:

```powershell
python 4tictactoe/main.py --experiment 1 --episodes 500000 --fixed-epsilon-grid 0.1,0.2,0.3,0.5
```

For each configuration, the saved agent is selected from the best run, not just
the final seed. Selection prioritizes draw rate against the search baseline,
then policy agreement, then win rate against the random opponent.

Interactive play after training:

```powershell
python 4tictactoe/main.py --play --agent-path results/agent_exp2_selfplay.pkl
```

## Note

The search baseline is depth-limited for tractability. Its agreement score is a
useful comparative metric, but it is not the same as exact 3x3 Minimax
optimality.
