"""
evaluation.py
-------------
Evaluation functions and plotting utilities.

Metrics implemented
-------------------
1. evaluate_vs_random       – win/draw/loss rate against a random opponent
2. evaluate_vs_minimax      – performance against depth-limited search baseline
3. compute_policy_optimality – fraction of moves matching the search baseline
4. plot_training_curves     – visualise training histories
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple

from .environment import BOARD_SIZE, WIN_LENGTH, TicTacToeEnv
from .agents import QLearningAgent, RandomAgent, MinimaxAgent


# =========================================================================== #
#  Evaluation helpers                                                           #
# =========================================================================== #

def evaluate_vs_random(
    agent: QLearningAgent,
    episodes: int = 1_000,
    agent_player: int = 1,
) -> Tuple[int, int, int]:
    env = TicTacToeEnv()
    opponent = RandomAgent(player=3 - agent_player)

    wins = draws = losses = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == agent_player:
                available = env.get_available_actions(state)
                action = agent.select_action(state, available, greedy=True)
            else:
                available = env.get_available_actions(state)
                action = opponent.select_action(state, available)

            state, _, done, info = env.step(action)

        winner = info["winner"]
        if winner == agent_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def evaluate_vs_minimax(
    agent: QLearningAgent,
    episodes: int = 200,
    agent_player: int = 1,
) -> Tuple[int, int, int]:
    env = TicTacToeEnv()
    minimax = MinimaxAgent(player=3 - agent_player)

    wins = draws = losses = 0

    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if env.current_player == agent_player:
                available = env.get_available_actions(state)
                action = agent.select_action(state, available, greedy=True)
            else:
                available = env.get_available_actions(state)
                action = minimax.select_action(state, available)

            state, _, done, info = env.step(action)

        winner = info["winner"]
        if winner == agent_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def compute_policy_optimality(
    agent: QLearningAgent,
    num_states: int = 500,
    agent_player: int = 1,
) -> float:
    env = TicTacToeEnv()
    minimax = MinimaxAgent(player=agent_player)

    agree = 0
    total = 0

    for _ in range(num_states):
        state = env.reset()
        done = False

        # Random walk to reach a diverse board state
        num_random_moves = np.random.randint(0, BOARD_SIZE + 4)
        for _ in range(num_random_moves):
            if done:
                break
            avail = env.get_available_actions(state)
            if not avail:
                break
            act = np.random.choice(avail)
            state, _, done, _ = env.step(act)

        if done:
            continue

        if env.current_player != agent_player:
            continue

        available = env.get_available_actions(state)
        if not available:
            continue

        agent_action = agent.select_action(state, available, greedy=True)

        # Use score-based comparison: the agent matches the search baseline
        # if its chosen action has the same score as the best available action.
        agent_action_score = minimax.score_action(state, agent_action)
        best_possible_score = minimax.best_score(state, available)

        if agent_action_score == best_possible_score:
            agree += 1
        total += 1

    return agree / total if total > 0 else 0.0


# =========================================================================== #
#  Plotting                                                                     #
# =========================================================================== #

def plot_training_curves(
    histories: Dict[str, Dict],
    save_dir: str = "results",
    filename: str = "training_curves.png",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    axes_titles = [
        ("Win Rate vs Random",    "win_rate_vs_random",      "Win Rate"),
        ("Draw Rate vs Random",   "draw_rate_vs_random",     "Draw Rate"),
        ("Average Reward",        "avg_reward",               "Avg Reward"),
        ("Epsilon (ε) Decay",     "epsilon",                  "ε"),
        ("Exploration Fraction",  "random_action_fraction",   "Fraction Random"),
        ("TD Error (|δ|)",        "td_error",                  "|TD Error|"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))  # type: ignore

    for idx, (title, key, ylabel) in enumerate(axes_titles):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])

        for (label, hist), color in zip(histories.items(), colors):
            if key in hist and hist[key]:
                ax.plot(hist["episode"], hist[key], label=label,
                        color=color, linewidth=1.8, alpha=0.9)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Q-Learning Training Curves - {BOARD_SIZE}x{BOARD_SIZE} Connect-{WIN_LENGTH}",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved -> {out_path}")


def plot_policy_optimality_bar(
    scores: Dict[str, float],
    save_dir: str = "results",
    filename: str = "policy_optimality.png",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    labels = list(scores.keys())
    values = [scores[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=plt.cm.tab10(np.linspace(0, 0.6, len(labels))),  # type: ignore
                  edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Policy Agreement vs. Search Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("% Moves Matching Search", fontsize=11)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved -> {out_path}")


def print_final_summary(
    label: str,
    agent: QLearningAgent,
    eval_episodes: int = 1_000,
    optimality_states: int = 500,
    verbose: bool = True,
) -> Dict:
    w_r, d_r, l_r = evaluate_vs_random(agent, episodes=eval_episodes)
    w_m, d_m, l_m = evaluate_vs_minimax(agent, episodes=min(eval_episodes, 300))
    opt = compute_policy_optimality(agent, num_states=optimality_states)

    n = eval_episodes
    nm = min(eval_episodes, 300)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  vs Random   : W={w_r/n:.2%}  D={d_r/n:.2%}  L={l_r/n:.2%}")
        print(f"  vs Search   : W={w_m/nm:.2%}  D={d_m/nm:.2%}  L={l_m/nm:.2%}")
        print(f"  Policy Agr. : {opt:.2%}  (agreement with search baseline)")
        print(f"  Q-table size: {len(agent.q_table)} entries")
        print(f"{'='*60}\n")

    return {
        "win_vs_random": w_r / n,
        "draw_vs_random": d_r / n,
        "loss_vs_random": l_r / n,
        "win_vs_minimax": w_m / nm,
        "draw_vs_minimax": d_m / nm,
        "loss_vs_minimax": l_m / nm,
        "policy_optimality": opt,
        "q_table_size": len(agent.q_table),
    }
