"""
evaluation.py
-------------
Evaluation functions and plotting utilities.

Metrics implemented
-------------------
1. evaluate_vs_random       – win/draw/loss rate against a random opponent
2. evaluate_vs_minimax      – win/draw/loss vs Minimax (optional agent as P1 or P2)
3. compute_policy_optimality – fraction of moves matching Minimax's choice
4. plot_training_curves     – visualise training histories
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from typing import Dict, List, Tuple

from .environment import TicTacToeEnv
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
    Play ``episodes`` games from empty board: ``agent`` plays as ``agent_player``
    (1 = X first, 2 = O second); Minimax is the opponent.
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
    minimax_eval = MinimaxAgent(player=agent_player)  # separate cache

    agree = 0
    total = 0

    for _ in range(num_states):
        state = env.reset()
        done = False

        # Random walk to reach a diverse board state
        num_random_moves = np.random.randint(0, 5)
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

        # Use score-based comparison: agent is "optimal" if its chosen
        # action has the same minimax score as the best possible action.
        # This correctly handles cases where multiple symmetric moves are
        # equally optimal.
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

    fig.suptitle("Q-Learning Training Curves – Tic-Tac-Toe",
                 fontsize=14, fontweight="bold", y=1.01)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved → {out_path}")


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

    ax.set_title("Policy Optimality vs. Minimax Baseline",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("% Moves Matching Minimax", fontsize=11)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plots] Saved → {out_path}")


def _save_png_and_pdf(fig, save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)

    stem, ext = os.path.splitext(filename)
    if ext.lower() == ".png":
        pdf_path = os.path.join(save_dir, f"{stem}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"[Plots] Saved → {pdf_path}")

    print(f"[Plots] Saved → {out_path}")


def plot_compact_training_curves(
    histories: Dict[str, List[Dict]],
    save_dir: str = "results",
    filename: str = "exp2_paradigms_compact.png",
) -> None:
    """Plot the report-friendly 1x3 Exp. 2 curve summary over all seeds."""
    if not histories:
        return

    panels = [
        ("Win Rate vs Random", "win_rate_vs_random", "Win Rate", (0.60, 1.02)),
        ("Draw Rate vs Random", "draw_rate_vs_random", "Draw Rate", (-0.002, 0.085)),
        ("TD Error (|δ|)", "td_error", "|TD Error|", (0.035, 0.32)),
    ]
    colors = {
        "vs Random": "#1f77b4",
        "Self-play": "#17becf",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.2),
        sharex=True,
        gridspec_kw=dict(left=0.06, right=0.985, bottom=0.18, top=0.78, wspace=0.28),
    )
    fig.patch.set_facecolor("white")

    for ax, (title, key, ylabel, ylim) in zip(axes, panels):
        for label, runs in histories.items():
            valid_runs = [h for h in runs if key in h and h[key]]
            if not valid_runs:
                continue

            episodes = np.array(valid_runs[0]["episode"], dtype=float)
            vals = np.array([h[key] for h in valid_runs], dtype=float)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            color = colors.get(label, None)

            ax.plot(episodes, mean, label=label, color=color, linewidth=2.3)
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("Episodes", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, framealpha=0.9, loc="best")

    fig.suptitle("Exp 2: Training Paradigms (8 Seeds)", fontsize=16, fontweight="bold", y=0.95)
    fig.text(
        0.5,
        0.865,
        "Mean curves with ±1 std. shaded bands",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
    )

    _save_png_and_pdf(fig, save_dir, filename)
    plt.close(fig)


def plot_minimax_wdl_summary(
    exp1: Dict,
    exp2: Dict,
    exp3: Dict,
    save_dir: str = "results",
    filename: str = "minimax_wdl_slides.png",
) -> None:
    """Plot a compact win/draw/loss summary against minimax for Exp. 1--3."""
    colors = {
        "win": "#4CAF50",
        "draw": "#5B9BD5",
        "loss": "#E74C3C",
        "bg": "#FAFAFA",
        "grid": "#E0E0E0",
    }
    alpha = 0.92
    bar_w = 0.35
    pair_gap = 0.08
    group_gap = 0.45
    pad = 0.30

    def wdl(data: Dict, key: str) -> Tuple[np.ndarray, np.ndarray]:
        m = data[key]["mean"]
        p1 = np.array([m["win_vs_minimax"], m["draw_vs_minimax"], m["loss_vs_minimax"]])
        p2 = np.array([m["win_vs_minimax_p2"], m["draw_vs_minimax_p2"], m["loss_vs_minimax_p2"]])
        return p1, p2

    panels = [
        (
            [wdl(exp1, k) for k in ["ε-fixed", "ε-linear", "ε-exponential"]],
            ["Fixed ε", "Linear ε", "Exp ε"],
            "Exp 1 – Exploration Schedule\n(trained vs Random)",
        ),
        (
            [wdl(exp2, k) for k in ["vs Random", "Self-play"]],
            ["vs Random", "Self-Play"],
            "Exp 2 – Training Paradigm\n(linear ε)",
        ),
        (
            [wdl(exp3, k) for k in list(exp3.keys())],
            ["Fixed ε", "Linear ε", "Exp ε"],
            "Exp 3 – Schedule × Minimax Eval\n(trained vs Random)",
        ),
    ]

    def draw_stacked_bars(ax, wdl_list, bar_labels, title):
        group_span = bar_w + pair_gap + bar_w
        step = group_span + group_gap
        centres = [i * step for i in range(len(wdl_list))]
        pos_p1 = [c - pair_gap / 2 - bar_w for c in centres]
        pos_p2 = [c + pair_gap / 2 for c in centres]

        for i, (p1, p2) in enumerate(wdl_list):
            for pos, vals, hatch in [(pos_p1[i], p1, ""), (pos_p2[i], p2, "///")]:
                bottom = 0.0
                for val, col in zip(vals, [colors["win"], colors["draw"], colors["loss"]]):
                    if val > 0:
                        ax.bar(
                            pos,
                            val,
                            bar_w,
                            bottom=bottom,
                            color=col,
                            alpha=alpha,
                            edgecolor="white",
                            linewidth=1.2,
                            hatch=hatch,
                            zorder=3,
                        )
                        if val >= 0.08:
                            txt = ax.text(
                                pos + bar_w / 2,
                                bottom + val / 2,
                                f"{val:.0%}",
                                ha="center",
                                va="center",
                                fontsize=12,
                                fontweight="bold",
                                color="white",
                                zorder=5,
                                clip_on=False,
                            )
                            txt.set_path_effects([
                                pe.Stroke(linewidth=2.8, foreground="#333"),
                                pe.Normal(),
                            ])
                    bottom += val

        ax.set_xticks(centres)
        ax.set_xticklabels(bar_labels, fontsize=13, fontweight="bold")
        for i in range(len(wdl_list)):
            ax.text(pos_p1[i] + bar_w / 2, -0.068, "P1", ha="center", va="top", fontsize=10, color="#555")
            ax.text(pos_p2[i] + bar_w / 2, -0.068, "P2", ha="center", va="top", fontsize=10, color="#555")

        ax.set_xlim(pos_p1[0] - pad, pos_p2[-1] + bar_w + pad)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
        ax.yaxis.grid(True, color=colors["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#BBBBBB")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color="#222")

    fig = plt.figure(figsize=(15, 6.2), facecolor=colors["bg"])
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[3, 2, 3],
        left=0.06,
        right=0.97,
        top=0.83,
        bottom=0.20,
        wspace=0.40,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(colors["bg"])

    for ax, (wdl_list, labels, title) in zip(axes, panels):
        draw_stacked_bars(ax, wdl_list, labels, title)
    axes[0].set_ylabel("Rate vs Minimax", fontsize=13, color="#333")

    fig.legend(
        handles=[
            mpatches.Patch(color=colors["win"], alpha=alpha, label="Win"),
            mpatches.Patch(color=colors["draw"], alpha=alpha, label="Draw"),
            mpatches.Patch(color=colors["loss"], alpha=alpha, label="Loss"),
            mpatches.Patch(facecolor="#777", alpha=0.55, label="Player 1 (solid)"),
            mpatches.Patch(facecolor="#777", alpha=0.55, hatch="///", edgecolor="white", label="Player 2 (hatched)"),
        ],
        loc="lower center",
        ncol=5,
        fontsize=12,
        framealpha=0.9,
        edgecolor="#CCC",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    fig.suptitle(
        "Q-Learning vs Minimax  –  Win / Draw / Loss  (3×3 Tic-Tac-Toe)",
        fontsize=17,
        fontweight="bold",
        color="#1A1A2E",
        y=0.985,
    )

    _save_png_and_pdf(fig, save_dir, filename)
    plt.close(fig)


def print_final_summary(
    label: str,
    agent: QLearningAgent,
    eval_episodes: int = 1_000,
    optimality_states: int = 500,
    verbose: bool = True,
) -> Dict:
    w_r, d_r, l_r = evaluate_vs_random(agent, episodes=eval_episodes)
    n = eval_episodes
    nm = min(eval_episodes, 300)
    # P1 = agent as X (moves first); P2 = agent as O (Minimax moves first).
    w_m, d_m, l_m = evaluate_vs_minimax(agent, episodes=nm, agent_player=1)
    w_m2, d_m2, l_m2 = evaluate_vs_minimax(agent, episodes=nm, agent_player=2)
    opt = compute_policy_optimality(agent, num_states=optimality_states)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  vs Random   : W={w_r/n:.2%}  D={d_r/n:.2%}  L={l_r/n:.2%}")
        print(
            f"  vs Minimax (agent X, first): "
            f"W={w_m/nm:.2%}  D={d_m/nm:.2%}  L={l_m/nm:.2%}"
        )
        print(
            f"  vs Minimax (agent O, second): "
            f"W={w_m2/nm:.2%}  D={d_m2/nm:.2%}  L={l_m2/nm:.2%}"
        )
        print(f"  Policy Opt. : {opt:.2%}  (agreement with Minimax)")
        print(f"  Q-table size: {len(agent.q_table)} entries")
        print(f"{'='*60}\n")

    return {
        "win_vs_random": w_r / n,
        "draw_vs_random": d_r / n,
        "loss_vs_random": l_r / n,
        # Legacy keys: agent plays as P1 (X, first to move).
        "win_vs_minimax": w_m / nm,
        "draw_vs_minimax": d_m / nm,
        "loss_vs_minimax": l_m / nm,
        "win_vs_minimax_p2": w_m2 / nm,
        "draw_vs_minimax_p2": d_m2 / nm,
        "loss_vs_minimax_p2": l_m2 / nm,
        "policy_optimality": opt,
        "q_table_size": len(agent.q_table),
    }
