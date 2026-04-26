# Tic-Tac-Toe Q-Learning Project

This repository implements a Tic-Tac-Toe environment and Q-learning agents using a Python package layout.

## Project structure

- `main.py` — entry point for running experiments and interactive play.
- `src/` — package containing core modules:
  - `src/environment.py` — Tic-Tac-Toe environment and helper functions.
  - `src/agents.py` — Q-learning, random, and Minimax agents.
  - `src/training.py` — training routines for random-opponent and self-play.
  - `src/evaluation.py` — evaluation metrics and plotting utilities.

## Requirements

- Python 3.8+ recommended
- `numpy`
- `matplotlib`

Install dependencies with pip:

```bash
pip install numpy matplotlib
```

## Run experiments

From the project root directory, run:

```bash
python main.py
```

This runs all experiments and saves outputs in the `results/` folder.

### Optional arguments

```bash
python main.py --episodes 50000
python main.py --eval-interval 2000
python main.py --experiment 1
python main.py --experiment 2
python main.py --experiment 3
python main.py --play --agent-path results/agent_exp2_vs_random.pkl
```

## Interactive play

Use a saved agent file to play against the trained Q-learning agent:

```bash
python main.py --play --agent-path results/agent_exp2_vs_random.pkl
```

The human player plays as O (player 2) and the agent plays as X (player 1).

## Notes

- The project is already structured as a package with `src/` and `src/__init__.py`.
- If you modify file paths, ensure the package imports remain correct.
