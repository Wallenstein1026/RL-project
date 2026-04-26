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
#  Experiment 1 – Exploration Schedule Ablation                                #
# =========================================================================== #

def experiment_1(total_episodes: int, eval_interval: int) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Exploration Schedule Ablation (vs Random Opponent)")
    print("=" * 70)

    schedules = ["fixed", "linear", "exponential"]
    histories = {}
    summaries = {}
    opt_scores = {}

    for sched in schedules:
        print(f"\n--- Schedule: {sched} ---")
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
            verbose=True,
        )
        histories[f"ε-{sched}"] = hist

        label = f"ε-{sched} (vs Random)"
        summary = print_final_summary(label, agent)
        summaries[label] = summary
        opt_scores[f"ε-{sched}"] = summary["policy_optimality"]

        agent.save(f"results/agent_exp1_{sched}.pkl")

    plot_training_curves(histories, save_dir="results",
                         filename="exp1_schedules.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp1_optimality.png")

    with open("results/exp1_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n[Exp 1] Done. Figures saved to results/")


# =========================================================================== #
#  Experiment 2 – Training Paradigm Comparison                                  #
# =========================================================================== #

def experiment_2(total_episodes: int, eval_interval: int) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Training Paradigm – vs Random vs Self-Play")
    print("=" * 70)

    histories = {}
    opt_scores = {}

    # --- vs Random ---
    print("\n--- Paradigm: vs Random (linear ε) ---")
    agent_rand = QLearningAgent(
        player=1, schedule="linear",
        total_episodes=total_episodes, **DEFAULT_HP
    )
    h_rand = train_vs_random(
        agent_rand,
        total_episodes=total_episodes,
        eval_interval=eval_interval,
        verbose=True,
    )
    histories["vs Random"] = h_rand
    s_rand = print_final_summary("vs Random", agent_rand)
    opt_scores["vs Random"] = s_rand["policy_optimality"]
    agent_rand.save("results/agent_exp2_vs_random.pkl")

    # --- Self-play ---
    print("\n--- Paradigm: Self-play (linear ε) ---")
    agent_sp1 = QLearningAgent(
        player=1, schedule="linear",
        total_episodes=total_episodes, **DEFAULT_HP
    )
    agent_sp2 = QLearningAgent(
        player=2, schedule="linear",
        total_episodes=total_episodes, **DEFAULT_HP
    )
    h_sp = train_selfplay(
        agent_sp1, agent_sp2,
        total_episodes=total_episodes,
        eval_interval=eval_interval,
        verbose=True,
    )
    histories["Self-play"] = h_sp
    s_sp = print_final_summary("Self-play (agent1)", agent_sp1)
    opt_scores["Self-play"] = s_sp["policy_optimality"]
    agent_sp1.save("results/agent_exp2_selfplay.pkl")

    plot_training_curves(histories, save_dir="results",
                         filename="exp2_paradigms.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp2_optimality.png")

    summaries = {"vs_random": s_rand, "selfplay": s_sp}
    with open("results/exp2_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n[Exp 2] Done. Figures saved to results/")


# =========================================================================== #
#  Experiment 3 – Comprehensive Policy Optimality                               #
# =========================================================================== #

def experiment_3(total_episodes: int, eval_interval: int) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Policy Optimality vs Minimax (All Configurations)")
    print("=" * 70)

    configs = [
        ("vs_Random + fixed_ε",       dict(schedule="fixed")),
        ("vs_Random + linear_ε",      dict(schedule="linear")),
        ("vs_Random + exp_ε",         dict(schedule="exponential")),
    ]

    opt_scores = {}
    minimax_results = {}

    for label, extra in configs:
        print(f"\n  Training: {label}")
        agent = QLearningAgent(
            player=1,
            total_episodes=total_episodes,
            **{**DEFAULT_HP, **extra},
        )
        train_vs_random(agent, total_episodes=total_episodes,
                        eval_interval=eval_interval, verbose=False)

        opt = compute_policy_optimality(agent, num_states=800)
        w, d, l = evaluate_vs_minimax(agent, episodes=300)
        opt_scores[label] = opt
        minimax_results[label] = {
            "win": w / 300, "draw": d / 300, "loss": l / 300,
            "policy_optimality": opt,
        }
        print(f"    Opt={opt:.2%}  W={w/300:.2%}  D={d/300:.2%}  L={l/300:.2%}")

    plot_policy_optimality_bar(opt_scores, save_dir="results",
                               filename="exp3_optimality_all.png")

    with open("results/exp3_minimax_results.json", "w") as f:
        json.dump(minimax_results, f, indent=2)

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

    if args.experiment is None or args.experiment == 1:
        experiment_1(ep, iv)
    if args.experiment is None or args.experiment == 2:
        experiment_2(ep, iv)
    if args.experiment is None or args.experiment == 3:
        experiment_3(ep, iv)

    print("\n✅  All experiments complete. Results saved in results/")


if __name__ == "__main__":
    main()