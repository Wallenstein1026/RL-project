"""
main.py
-------
Entry point for the Tic-Tac-Toe Q-Learning project.

Runs three sets of experiments and generates all result plots:

  Experiment 1 – Exploration Schedule Ablation
      Agent trained vs random opponent with fixed / linear / exponential ε.

  Experiment 2 – Training Paradigm Comparison
      Agent trained vs random opponent  VS  self-play.

  Experiment 3 – Final Policy Optimality
      Measure each trained agent's agreement with the Minimax baseline.

Usage
-----
    python main.py                          # run all experiments (default)
    python main.py --episodes 50000         # shorter run for quick testing
    python main.py --experiment 1           # run only experiment 1
    python main.py --num-runs 5             # 5 runs per config (statistics)
    python main.py --seed 42                # base seed for reproducibility
    python main.py --play                   # play interactively vs trained agent
"""

import argparse
import json
import os
import sys

import numpy as np

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

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
from src.environment import TicTacToeEnv


# =========================================================================== #
#  Hyperparameters (shared across experiments unless overridden)                #
# =========================================================================== #

DEFAULT_HP = dict(
    alpha=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
)


# =========================================================================== #
#  Statistics helpers                                                           #
# =========================================================================== #

def _stats_str(values: list) -> str:
    """Format mean ± std for a list of floats."""
    if len(values) <= 1:
        return f"{values[0]:.2%}" if values else "N/A"
    m = np.mean(values)
    s = np.std(values)
    return f"{m:.2%} ± {s:.2%}"


def _print_run_summary(label: str, values: dict):
    """Print a summary table for one config across multiple runs."""
    print(f"\n  {label}")
    print(f"  {'─' * 60}")
    for metric, vals in values.items():
        print(f"    {metric:30s}: {_stats_str(vals)}")


# =========================================================================== #
#  Experiment 1 – Exploration Schedule Ablation                                #
# =========================================================================== #

def experiment_1(total_episodes: int, eval_interval: int,
                 num_runs: int = 3, base_seed: int = 42) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Exploration Schedule Ablation (vs Random Opponent)")
    print(f"  Runs per config: {num_runs}  |  Base seed: {base_seed}")
    print("=" * 70)

    schedules = ["fixed", "linear", "exponential"]
    all_histories: dict = {}      # last-run history for plotting
    all_summaries: dict = {}      # multi-run aggregated results
    opt_scores: dict = {}         # mean optimality for bar chart

    for sched in schedules:
        print(f"\n--- Schedule: {sched} ---")

        run_metrics = {
            "win_vs_random": [], "draw_vs_random": [], "loss_vs_random": [],
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [], "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                schedule=sched,
                total_episodes=total_episodes,
                **DEFAULT_HP,
            )

            hist = train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
            )

            if run == num_runs - 1:
                all_histories[f"ε-{sched}"] = hist

            label = f"ε-{sched} (vs Random)"
            summary = print_final_summary(label, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        # Print aggregated stats
        _print_run_summary(f"ε-{sched} (vs Random) — {num_runs} runs", run_metrics)
        opt_scores[f"ε-{sched}"] = np.mean(run_metrics["policy_optimality"])

        all_summaries[f"ε-{sched}"] = {
            "num_runs": num_runs,
            "base_seed": base_seed,
            "seeds": [base_seed + r for r in range(num_runs)],
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std":  {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics}
                           for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results",
                         filename="exp1_schedules.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp1_optimality.png")

    # Save last-run agent for interactive play
    # (use the last schedule's agent from the last run — already saved above)

    with open("results/exp1_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n[Exp 1] Done. Figures saved to results/")


# =========================================================================== #
#  Experiment 2 – Training Paradigm Comparison                                  #
# =========================================================================== #

def experiment_2(total_episodes: int, eval_interval: int,
                 num_runs: int = 3, base_seed: int = 42) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Training Paradigm – vs Random vs Self-Play")
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
        print(f"\n--- Paradigm: {paradigm} (linear ε) ---")

        run_metrics = {
            "win_vs_random": [], "draw_vs_random": [], "loss_vs_random": [],
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [], "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + 100 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            if mode == "train_vs_random":
                agent = QLearningAgent(
                    player=1, schedule="linear",
                    total_episodes=total_episodes, **DEFAULT_HP
                )
                hist = train_vs_random(
                    agent, total_episodes=total_episodes,
                    eval_interval=eval_interval, verbose=False, seed=seed,
                )
                label = "vs Random"
            else:
                agent_sp1 = QLearningAgent(
                    player=1, schedule="linear",
                    total_episodes=total_episodes, **DEFAULT_HP
                )
                agent_sp2 = QLearningAgent(
                    player=2, schedule="linear",
                    total_episodes=total_episodes, **DEFAULT_HP
                )
                hist = train_selfplay(
                    agent_sp1, agent_sp2,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval, verbose=False, seed=seed,
                )
                agent = agent_sp1
                label = "Self-play"

            if run == num_runs - 1:
                all_histories[paradigm] = hist

            summary = print_final_summary(
                f"{label} (run {run+1})", agent, verbose=False
            )
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"{paradigm} — {num_runs} runs", run_metrics)
        opt_scores[paradigm] = np.mean(run_metrics["policy_optimality"])

        all_summaries[paradigm] = {
            "num_runs": num_runs,
            "base_seed": base_seed,
            "seeds": [base_seed + 100 + r for r in range(num_runs)],
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
                 num_runs: int = 3, base_seed: int = 42) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Policy Optimality vs Minimax (All Configurations)")
    print(f"  Runs per config: {num_runs}  |  Base seed: {base_seed}")
    print("=" * 70)

    configs = [
        ("vs_Random + fixed_ε",       dict(schedule="fixed")),
        ("vs_Random + linear_ε",      dict(schedule="linear")),
        ("vs_Random + exp_ε",         dict(schedule="exponential")),
    ]

    opt_scores: dict = {}
    all_results: dict = {}

    for idx, (label, extra) in enumerate(configs):
        print(f"\n  Training: {label}")

        run_metrics = {
            "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
            "policy_optimality": [],
        }

        for run in range(num_runs):
            seed = base_seed + 200 + idx * 10 + run
            print(f"    [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                total_episodes=total_episodes,
                **{**DEFAULT_HP, **extra},
            )
            train_vs_random(agent, total_episodes=total_episodes,
                            eval_interval=eval_interval, verbose=False,
                            seed=seed)

            opt = compute_policy_optimality(agent, num_states=800)
            w, d, l = evaluate_vs_minimax(agent, episodes=300)

            run_metrics["policy_optimality"].append(opt)
            run_metrics["win_vs_minimax"].append(w / 300)
            run_metrics["draw_vs_minimax"].append(d / 300)
            run_metrics["loss_vs_minimax"].append(l / 300)

            print(f"      Opt={opt:.2%}  W={w/300:.2%}  D={d/300:.2%}  L={l/300:.2%}")

        opt_scores[label] = np.mean(run_metrics["policy_optimality"])
        all_results[label] = {
            "num_runs": num_runs,
            "seeds": [base_seed + 200 + idx * 10 + r for r in range(num_runs)],
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
    print("0 | 1 | 2\n3 | 4 | 5\n6 | 7 | 8\n")

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
    p = argparse.ArgumentParser(description="Q-Learning Tic-Tac-Toe")
    p.add_argument("--episodes", type=int, default=100_000,
                   help="Total training episodes (default: 100000)")
    p.add_argument("--eval-interval", type=int, default=2_000,
                   help="Evaluation checkpoint interval (default: 2000)")
    p.add_argument("--experiment", type=int, choices=[1, 2, 3], default=None,
                   help="Run only experiment 1, 2, or 3 (default: all)")
    p.add_argument("--num-runs", type=int, default=3,
                   help="Number of runs per config for statistics (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed for reproducibility (default: 42)")
    p.add_argument("--play", action="store_true",
                   help="Play interactively against a saved agent")
    p.add_argument("--agent-path", type=str,
                   default="results/agent_exp2_vs_random.pkl",
                   help="Path to agent pickle for --play mode")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    if args.play:
        play_vs_agent(args.agent_path)
        return

    ep = args.episodes
    iv = args.eval_interval
    nr = args.num_runs
    seed = args.seed

    if args.experiment is None or args.experiment == 1:
        experiment_1(ep, iv, num_runs=nr, base_seed=seed)
    if args.experiment is None or args.experiment == 2:
        experiment_2(ep, iv, num_runs=nr, base_seed=seed)
    if args.experiment is None or args.experiment == 3:
        experiment_3(ep, iv, num_runs=nr, base_seed=seed)

    print(f"\n✅  All experiments complete. Results saved in results/")


if __name__ == "__main__":
    main()
