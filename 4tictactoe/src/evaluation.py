"""
evaluation.py
-------------
Evaluation functions and plotting utilities.

Metrics implemented
-------------------
1. evaluate_vs_random       – win/draw/loss rate against a random opponent
2. evaluate_vs_minimax      – performance against depth-limited search baseline
                              with the agent as P1 or P2
3. compute_policy_optimality – fraction of moves matching the search baseline
4. plot_training_curves     – visualise training histories
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
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
    """
    Play games from an empty board with the agent as X/P1 or O/P2.

    ``agent_player=1`` keeps the legacy first-player metric. ``agent_player=2``
    is the second-player check used for parity with the 3x3 experiments.
    """
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


def plot_improvements_summary(
    symmetry_results: Dict,
    exp1_results: Dict,
    ucb_results: Dict,
    save_dir: str = "results",
    filename: str = "improvements_slides.png",
) -> None:
    """Plot the report summary for D4 symmetry and UCB improvements."""
    colors = {
        "base": "#7F8C8D",
        "sym": "#2ECC71",
        "ucb": "#3498DB",
        "bg": "#FAFAFA",
        "grid": "#E8E8E8",
    }
    alpha = 0.88
    bar_w = 0.32
    err_kw = dict(capsize=5, capthick=1.8, elinewidth=1.8, ecolor="#444")

    base_opt_m = symmetry_results["baseline_fixed_0.5"]["mean"]["policy_optimality"]
    base_opt_s = symmetry_results["baseline_fixed_0.5"]["std"]["policy_optimality"]
    sym_opt_m = symmetry_results["symmetry_fixed_0.5"]["mean"]["policy_optimality"]
    sym_opt_s = symmetry_results["symmetry_fixed_0.5"]["std"]["policy_optimality"]

    base_draw_m = exp1_results["linear_eps"]["mean"]["draw_vs_minimax"]
    base_draw_s = exp1_results["linear_eps"]["std"]["draw_vs_minimax"]
    ucb_draw_m = ucb_results["vs_Random + ucb"]["mean"]["draw_vs_minimax"]
    ucb_draw_s = ucb_results["vs_Random + ucb"]["std"]["draw_vs_minimax"]

    def bar_panel(
        ax,
        left_val,
        left_err,
        right_val,
        right_err,
        left_label,
        right_label,
        left_col,
        right_col,
        title,
        ylabel,
        ymax,
    ):
        positions = [0.0, bar_w + 0.18]
        vals = [left_val, right_val]
        errs = [left_err, right_err]
        cols = [left_col, right_col]
        labels = [left_label, right_label]

        for pos, val, err, col in zip(positions, vals, errs, cols):
            ax.bar(
                pos,
                val,
                bar_w,
                color=col,
                alpha=alpha,
                edgecolor="white",
                linewidth=1.4,
                zorder=3,
                align="edge",
            )
            ax.errorbar(pos + bar_w / 2, val, yerr=err, fmt="none", zorder=4, **err_kw)
            if val >= 0.04:
                txt = ax.text(
                    pos + bar_w / 2,
                    val / 2,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white",
                    zorder=5,
                )
                txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="#333"), pe.Normal()])

        delta = right_val - left_val
        sign = "+" if delta >= 0 else ""
        mid_x = positions[0] + bar_w + 0.09
        mid_y = max(left_val, right_val) + errs[1 if right_val > left_val else 0] + 0.03
        ax.annotate(
            f"{sign}{delta:.0%}",
            xy=(mid_x, mid_y),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#E74C3C" if delta >= 0 else "#666",
            zorder=5,
        )

        ax.set_xticks([p + bar_w / 2 for p in positions])
        ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.18, positions[-1] + bar_w + 0.18)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.grid(True, color=colors["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#BBBBBB")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#222")
        ax.set_ylabel(ylabel, fontsize=11, color="#444")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 6.0),
        facecolor=colors["bg"],
        gridspec_kw=dict(wspace=0.42, top=0.82, bottom=0.18, left=0.09, right=0.97),
    )
    ax1.set_facecolor(colors["bg"])
    ax2.set_facecolor(colors["bg"])

    bar_panel(
        ax1,
        base_opt_m,
        base_opt_s,
        sym_opt_m,
        sym_opt_s,
        "Baseline\n(ε-greedy)",
        "Symmetry\nAugmentation",
        colors["base"],
        colors["sym"],
        "Policy Optimality\n(4×4, fixed ε=0.5)",
        "Policy Optimality",
        0.60,
    )
    bar_panel(
        ax2,
        base_draw_m,
        base_draw_s,
        ucb_draw_m,
        ucb_draw_s,
        "Baseline\n(linear ε)",
        "UCB\n(c=0.5)",
        colors["base"],
        colors["ucb"],
        "Draw Rate vs Minimax\n(4×4, vs Random)",
        "Draw Rate vs Minimax",
        0.65,
    )

    fig.suptitle(
        "4×4 Tic-Tac-Toe  –  Effect of Symmetry Augmentation and UCB Exploration",
        fontsize=14,
        fontweight="bold",
        color="#1A1A2E",
        y=0.97,
    )
    fig.legend(
        handles=[
            mpatches.Patch(color=colors["base"], alpha=alpha, label="Baseline (ε-greedy)"),
            mpatches.Patch(color=colors["sym"], alpha=alpha, label="+ Symmetry Augmentation"),
            mpatches.Patch(color=colors["ucb"], alpha=alpha, label="UCB Exploration (c=0.5)"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=10.5,
        framealpha=0.9,
        edgecolor="#CCC",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=colors["bg"])
    stem, ext = os.path.splitext(filename)
    if ext.lower() == ".png":
        pdf_path = os.path.join(save_dir, f"{stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=colors["bg"])
        print(f"[Plots] Saved -> {pdf_path}")
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
    nm = min(eval_episodes, 300)
    w_m, d_m, l_m = evaluate_vs_minimax(agent, episodes=nm, agent_player=1)
    w_m2, d_m2, l_m2 = evaluate_vs_minimax(agent, episodes=nm, agent_player=2)
    opt = compute_policy_optimality(agent, num_states=optimality_states)

    n = eval_episodes

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  vs Random   : W={w_r/n:.2%}  D={d_r/n:.2%}  L={l_r/n:.2%}")
        print(
            f"  vs Search X : W={w_m/nm:.2%}  D={d_m/nm:.2%}  L={l_m/nm:.2%}"
        )
        print(
            f"  vs Search O : W={w_m2/nm:.2%}  D={d_m2/nm:.2%}  L={l_m2/nm:.2%}"
        )
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
        "win_vs_minimax_p2": w_m2 / nm,
        "draw_vs_minimax_p2": d_m2 / nm,
        "loss_vs_minimax_p2": l_m2 / nm,
        "policy_optimality": opt,
        "q_table_size": len(agent.q_table),
    }
