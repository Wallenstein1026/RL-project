"""
main.py
-------
Entry point for the 4x4 connect-4 Tic-Tac-Toe Q-Learning project.

Runs three sets of experiments and generates all result plots:

  Experiment 1 – Exploration Schedule Ablation
      Agent trained vs random opponent with fixed / linear / exponential ε.

  Experiment 2 – Training Paradigm Comparison
      Agent trained vs random opponent  VS  self-play.

  Experiment 3 – Search-Baseline Policy Agreement
      Measure each trained agent's agreement with a depth-limited search baseline.

Usage
-----
    python main.py                          # run all experiments (default)
    python main.py --episodes 50000         # shorter run for quick testing
    python main.py --experiment 1           # run only experiment 1
    python main.py --num-runs 5             # 5 runs per config (statistics)
    python main.py --seed 42                # base seed for reproducibility
    python main.py --fixed-epsilon-grid 0.1,0.2,0.3,0.5
    python main.py --play                   # play interactively vs trained agent
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(__file__)

# Make sure this project root is on the path
sys.path.insert(0, PROJECT_DIR)

from src.agents import QLearningAgent
from src.training import train_vs_random, train_selfplay
from src.evaluation import (
    plot_training_curves,
    plot_policy_optimality_bar,
    print_final_summary,
    evaluate_vs_random,
    evaluate_vs_minimax,
    compute_policy_optimality,
)
from src.environment import BOARD_CELLS, BOARD_SIZE, WIN_LENGTH, TicTacToeEnv


# =========================================================================== #
#  Hyperparameters (shared across experiments unless overridden)                #
# =========================================================================== #

DEFAULT_HP = dict(
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
)

DEFAULT_EPISODES = 500_000
DEFAULT_FIXED_EPSILON = 0.30
DEFAULT_FIXED_EPSILON_GRID = [0.10, 0.20, 0.30, 0.50]


def _hp_for_schedule(schedule: str, fixed_epsilon: float = DEFAULT_FIXED_EPSILON) -> dict:
    """Return hyperparameters for one exploration schedule."""
    hp = {**DEFAULT_HP, "schedule": schedule}
    if schedule == "fixed":
        hp["epsilon_start"] = fixed_epsilon
        hp["epsilon_end"] = fixed_epsilon
    return hp


def _format_eps(value: float) -> str:
    """Format epsilon values for labels and filenames."""
    return f"{value:g}".replace(".", "p")


def _parse_float_grid(value: str) -> list:
    """Parse a comma-separated float grid from the CLI."""
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("fixed epsilon grid cannot be empty")
    for eps in grid:
        if not 0.0 <= eps <= 1.0:
            raise ValueError("all fixed epsilon grid values must be between 0 and 1")
    return grid


def _schedule_configs(fixed_epsilon_grid: list) -> list:
    """Return schedule configs, including multiple fixed-epsilon variants."""
    configs = [
        (f"fixed_eps_{eps:g}", "fixed", f"fixed_{_format_eps(eps)}", eps)
        for eps in fixed_epsilon_grid
    ]
    configs.extend([
        ("linear_eps", "linear", "linear", DEFAULT_FIXED_EPSILON),
        ("exp_eps", "exponential", "exponential", DEFAULT_FIXED_EPSILON),
    ])
    return configs


def _run_score(metrics: dict) -> tuple:
    """
    Rank agents for saving.

    Prefer agents that hold the search baseline to draws, then those with
    higher one-step policy agreement, then better random-opponent win rate.
    """
    return (
        metrics.get("draw_vs_minimax", 0.0),
        metrics.get("policy_optimality", 0.0),
        metrics.get("win_vs_random", 0.0),
        -metrics.get("loss_vs_minimax", 0.0),
    )


def _maybe_update_best(best, agent: QLearningAgent, metrics: dict,
                       run: int, seed: int, history=None) -> dict:
    """Return the better of the current best agent and a new run."""
    candidate = {
        "agent": agent,
        "metrics": metrics,
        "run_index": run,
        "seed": seed,
        "score": _run_score(metrics),
        "history": history,
    }
    if best is None or candidate["score"] > best["score"]:
        return candidate
    return best


def _best_run_metadata(best: dict) -> dict:
    """Serializable metadata for the selected best run."""
    return {
        "run_index": best["run_index"],
        "seed": best["seed"],
        "selection_score": list(best["score"]),
        "metrics": best["metrics"],
    }


# =========================================================================== #
#  Statistics helpers                                                           #
# =========================================================================== #

def _stats_str(values: list, as_percent: bool = True) -> str:
    """Format mean ± std for a list of numeric values."""
    if len(values) <= 1:
        if not values:
            return "N/A"
        return f"{values[0]:.2%}" if as_percent else f"{values[0]:.0f}"
    m = np.mean(values)
    s = np.std(values)
    return f"{m:.2%} ± {s:.2%}" if as_percent else f"{m:.0f} ± {s:.0f}"


def _print_run_summary(label: str, values: dict):
    """Print a summary table for one config across multiple runs."""
    print(f"\n  {label}")
    print(f"  {'-' * 60}")
    for metric, vals in values.items():
        print(f"    {metric:30s}: {_stats_str(vals, as_percent=(metric != 'q_table_size'))}")


# =========================================================================== #
#  Experiment 1 – Exploration Schedule Ablation                                #
# =========================================================================== #

def experiment_1(total_episodes: int, eval_interval: int,
                 num_runs: int = 3, base_seed: int = 42,
                 fixed_epsilon_grid: list = DEFAULT_FIXED_EPSILON_GRID) -> None:
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT 1: Exploration Schedule Ablation ({BOARD_SIZE}x{BOARD_SIZE}, connect {WIN_LENGTH})")
    print(f"  Runs per config: {num_runs}  |  Base seed: {base_seed}")
    print(f"  Fixed epsilon grid: {fixed_epsilon_grid}")
    print("=" * 70)

    configs = _schedule_configs(fixed_epsilon_grid)
    all_histories: dict = {}      # last-run history for plotting
    all_summaries: dict = {}      # multi-run aggregated results
    opt_scores: dict = {}         # mean optimality for bar chart

    for config_label, sched, save_name, fixed_epsilon in configs:
        print(f"\n--- Schedule: {config_label} ---")

        run_metrics = {
            "win_vs_random": [], "draw_vs_random": [], "loss_vs_random": [],
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [], "q_table_size": [],
        }
        best_run = None

        for run in range(num_runs):
            seed = base_seed + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                total_episodes=total_episodes,
                **_hp_for_schedule(sched, fixed_epsilon),
            )

            hist = train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
            )

            label = f"{config_label} (vs Random)"
            summary = print_final_summary(label, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])
            best_run = _maybe_update_best(
                best_run, agent, summary, run, seed, history=hist
            )

        # Print aggregated stats
        _print_run_summary(f"{config_label} (vs Random) - {num_runs} runs", run_metrics)
        opt_scores[config_label] = np.mean(run_metrics["policy_optimality"])
        all_histories[config_label] = best_run["history"]
        best_run["agent"].save(f"results/agent_exp1_{save_name}.pkl")

        all_summaries[config_label] = {
            "board_size": BOARD_SIZE,
            "win_length": WIN_LENGTH,
            "total_episodes": total_episodes,
            "eval_interval": eval_interval,
            "fixed_epsilon_grid": fixed_epsilon_grid,
            "fixed_epsilon": fixed_epsilon,
            "hyperparameters": _hp_for_schedule(sched, fixed_epsilon),
            "num_runs": num_runs,
            "base_seed": base_seed,
            "seeds": [base_seed + r for r in range(num_runs)],
            "best_run": _best_run_metadata(best_run),
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std":  {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics}
                           for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results",
                         filename="exp1_schedules.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp1_optimality.png")

    with open("results/exp1_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n[Exp 1] Done. Figures saved to results/")


# =========================================================================== #
#  Experiment 2 – Training Paradigm Comparison                                  #
# =========================================================================== #

def experiment_2(total_episodes: int, eval_interval: int,
                 num_runs: int = 3, base_seed: int = 42,
                 fixed_epsilon: float = DEFAULT_FIXED_EPSILON) -> None:
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT 2: Training Paradigm ({BOARD_SIZE}x{BOARD_SIZE}, connect {WIN_LENGTH})")
    print(f"  Runs per config: {num_runs}  |  Base seed: {base_seed}")
    print("=" * 70)

    all_histories: dict = {}
    opt_scores: dict = {}
    all_summaries: dict = {}

    paradigms = {
        "vs Random": "train_vs_random",
        "Self-play": "train_selfplay",
    }

    for paradigm, mode in paradigms.items():
        print(f"\n--- Paradigm: {paradigm} (linear eps) ---")

        run_metrics = {
            "win_vs_random": [], "draw_vs_random": [], "loss_vs_random": [],
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [], "q_table_size": [],
        }
        best_run = None

        for run in range(num_runs):
            seed = base_seed + 100 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            if mode == "train_vs_random":
                agent = QLearningAgent(
                    player=1,
                    total_episodes=total_episodes,
                    **_hp_for_schedule("linear", fixed_epsilon),
                )
                hist = train_vs_random(
                    agent, total_episodes=total_episodes,
                    eval_interval=eval_interval, verbose=False, seed=seed,
                )
                label = "vs Random"
            else:
                agent_sp1 = QLearningAgent(
                    player=1,
                    total_episodes=total_episodes,
                    **_hp_for_schedule("linear", fixed_epsilon),
                )
                agent_sp2 = QLearningAgent(
                    player=2,
                    total_episodes=total_episodes,
                    **_hp_for_schedule("linear", fixed_epsilon),
                )
                hist = train_selfplay(
                    agent_sp1, agent_sp2,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval, verbose=False, seed=seed,
                )
                agent = agent_sp1
                label = "Self-play"

            summary = print_final_summary(
                f"{label} (run {run+1})", agent, verbose=False
            )
            for k in run_metrics:
                run_metrics[k].append(summary[k])
            best_run = _maybe_update_best(
                best_run, agent, summary, run, seed, history=hist
            )

        _print_run_summary(f"{paradigm} - {num_runs} runs", run_metrics)
        opt_scores[paradigm] = np.mean(run_metrics["policy_optimality"])
        all_histories[paradigm] = best_run["history"]
        if mode == "train_vs_random":
            best_run["agent"].save("results/agent_exp2_vs_random.pkl")
        else:
            best_run["agent"].save("results/agent_exp2_selfplay.pkl")

        all_summaries[paradigm] = {
            "board_size": BOARD_SIZE,
            "win_length": WIN_LENGTH,
            "total_episodes": total_episodes,
            "eval_interval": eval_interval,
            "fixed_epsilon": fixed_epsilon,
            "hyperparameters": _hp_for_schedule("linear", fixed_epsilon),
            "num_runs": num_runs,
            "base_seed": base_seed,
            "seeds": [base_seed + 100 + r for r in range(num_runs)],
            "best_run": _best_run_metadata(best_run),
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std":  {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics}
                           for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results",
                         filename="exp2_paradigms.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp2_optimality.png")

    with open("results/exp2_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n[Exp 2] Done. Figures saved to results/")


# =========================================================================== #
#  Experiment 3 – Comprehensive Policy Optimality                               #
# =========================================================================== #

def experiment_3(total_episodes: int, eval_interval: int,
                 num_runs: int = 3, base_seed: int = 42,
                 fixed_epsilon_grid: list = DEFAULT_FIXED_EPSILON_GRID) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Policy Agreement vs Search Baseline")
    print(f"  Runs per config: {num_runs}  |  Base seed: {base_seed}")
    print(f"  Fixed epsilon grid: {fixed_epsilon_grid}")
    print("=" * 70)

    configs = [
        (f"vs_Random + fixed_eps_{eps:g}", "fixed", f"fixed_{_format_eps(eps)}", eps)
        for eps in fixed_epsilon_grid
    ]
    configs.extend([
        ("vs_Random + linear_eps", "linear", "linear", DEFAULT_FIXED_EPSILON),
        ("vs_Random + exp_eps", "exponential", "exponential", DEFAULT_FIXED_EPSILON),
    ])

    opt_scores: dict = {}
    all_results: dict = {}

    for idx, (label, schedule, save_name, fixed_epsilon) in enumerate(configs):
        print(f"\n  Training: {label}")

        run_metrics = {
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [],
        }
        best_run = None

        for run in range(num_runs):
            seed = base_seed + 200 + idx * 10 + run
            print(f"    [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                total_episodes=total_episodes,
                **_hp_for_schedule(schedule, fixed_epsilon),
            )
            train_vs_random(agent, total_episodes=total_episodes,
                            eval_interval=eval_interval, verbose=False,
                            seed=seed)

            opt = compute_policy_optimality(agent, num_states=800)
            w, d, l = evaluate_vs_minimax(agent, episodes=300)
            summary = {
                "win_vs_minimax": w / 300,
                "draw_vs_minimax": d / 300,
                "loss_vs_minimax": l / 300,
                "policy_optimality": opt,
            }

            run_metrics["policy_optimality"].append(opt)
            run_metrics["win_vs_minimax"].append(w / 300)
            run_metrics["draw_vs_minimax"].append(d / 300)
            run_metrics["loss_vs_minimax"].append(l / 300)
            best_run = _maybe_update_best(best_run, agent, summary, run, seed)

            print(f"      Opt={opt:.2%}  W={w/300:.2%}  D={d/300:.2%}  L={l/300:.2%}")

        opt_scores[label] = np.mean(run_metrics["policy_optimality"])
        best_run["agent"].save(f"results/agent_exp3_{save_name}.pkl")
        all_results[label] = {
            "board_size": BOARD_SIZE,
            "win_length": WIN_LENGTH,
            "total_episodes": total_episodes,
            "eval_interval": eval_interval,
            "fixed_epsilon_grid": fixed_epsilon_grid,
            "fixed_epsilon": fixed_epsilon,
            "hyperparameters": _hp_for_schedule(schedule, fixed_epsilon),
            "num_runs": num_runs,
            "seeds": [base_seed + 200 + idx * 10 + r for r in range(num_runs)],
            "best_run": _best_run_metadata(best_run),
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std":  {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics}
                           for i in range(num_runs)],
        }
        _print_run_summary(label, run_metrics)

    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp3_optimality_all.png")

    with open("results/exp3_minimax_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Exp 3] Done. Figures saved to results/")


# =========================================================================== #
#  Interactive play                                                              #
# =========================================================================== #

def play_vs_agent(agent_path: str = "results/agent_exp2_vs_random.pkl") -> None:
    """Play a game against a trained Q-learning agent in the terminal."""
    from src.agents import QLearningAgent
    agent = QLearningAgent(player=1)
    agent.load(agent_path)

    env = TicTacToeEnv()
    state = env.reset()

    print("\nYou are O (player 2). The agent is X (player 1).")
    print("Board positions:")
    for row in range(BOARD_SIZE):
        print(" | ".join(
            f"{row * BOARD_SIZE + col:2d}" for col in range(BOARD_SIZE)
        ))
    print()

    while True:
        env.render(state)

        if env.current_player == 1:
            available = env.get_available_actions(state)
            action = agent.select_action(state, available, greedy=True)
            print(f"Agent plays: {action}")
        else:
            available = env.get_available_actions(state)
            print(f"Available: {available}")
            while True:
                try:
                    action = int(input("Your move: "))
                    if action in available:
                        break
                    print("Invalid. Try again.")
                except ValueError:
                    print("Enter a number.")

        state, _, done, info = env.step(action)

        if done:
            env.render(state)
            winner = info["winner"]
            if winner == 1:
                print("Agent wins!")
            elif winner == 2:
                print("You win!")
            else:
                print("Draw!")
            break


# =========================================================================== #
#  CLI                                                                           #
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description=f"Q-Learning {BOARD_SIZE}x{BOARD_SIZE} Connect-{WIN_LENGTH}"
    )
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                   help=f"Total training episodes (default: {DEFAULT_EPISODES})")
    p.add_argument("--eval-interval", type=int, default=2_000,
                   help="Evaluation checkpoint interval (default: 2000)")
    p.add_argument("--experiment", type=int, choices=[1, 2, 3], default=None,
                   help="Run only experiment 1, 2, or 3 (default: all)")
    p.add_argument("--num-runs", type=int, default=3,
                   help="Number of runs per config for statistics (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed for reproducibility (default: 42)")
    p.add_argument("--fixed-epsilon", type=float, default=DEFAULT_FIXED_EPSILON,
                   help=f"Fixed exploration rate for fixed schedule (default: {DEFAULT_FIXED_EPSILON})")
    p.add_argument("--fixed-epsilon-grid", type=str,
                   default=",".join(f"{eps:g}" for eps in DEFAULT_FIXED_EPSILON_GRID),
                   help="Comma-separated fixed epsilon grid for experiments 1 and 3 (default: 0.1,0.2,0.3,0.5)")
    p.add_argument("--play", action="store_true",
                   help="Play interactively against a saved agent")
    p.add_argument("--agent-path", type=str,
                   default="results/agent_exp2_vs_random.pkl",
                   help="Path to agent pickle for --play mode")
    return p.parse_args()


def main():
    os.chdir(PROJECT_DIR)
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    if args.play:
        play_vs_agent(args.agent_path)
        return

    ep = args.episodes
    iv = args.eval_interval
    nr = args.num_runs
    seed = args.seed
    fixed_epsilon = args.fixed_epsilon
    fixed_epsilon_grid = _parse_float_grid(args.fixed_epsilon_grid)

    if not 0.0 <= fixed_epsilon <= 1.0:
        raise ValueError("--fixed-epsilon must be between 0 and 1")

    if args.experiment is None or args.experiment == 1:
        experiment_1(ep, iv, num_runs=nr, base_seed=seed,
                     fixed_epsilon_grid=fixed_epsilon_grid)
    if args.experiment is None or args.experiment == 2:
        experiment_2(ep, iv, num_runs=nr, base_seed=seed,
                     fixed_epsilon=fixed_epsilon)
    if args.experiment is None or args.experiment == 3:
        experiment_3(ep, iv, num_runs=nr, base_seed=seed,
                     fixed_epsilon_grid=fixed_epsilon_grid)

    print("\n[Done] All experiments complete. Results saved in results/")


if __name__ == "__main__":
    main()
