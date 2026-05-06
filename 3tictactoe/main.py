"""
main.py
-------
Entry point for the Tic-Tac-Toe Q-Learning project.

Runs experiments including γ ablation, curriculum / mixed training,
TD vs MC, and optional potential-based shaping.

Usage
-----
    python main.py
    python main.py --experiment 4 --gamma 1.0
    python main.py --experiment 5 --potential-shaping
    python main.py --gamma 0.9 --experiment 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.agents import QLearningAgent
from src.training import (
    train_vs_random,
    train_selfplay,
    train_vs_random_then_selfplay,
    train_mixed_random_selfplay,
)
from src.evaluation import (
    plot_training_curves,
    plot_policy_optimality_bar,
    print_final_summary,
    evaluate_vs_random,
    evaluate_vs_minimax,
    compute_policy_optimality,
)
from src.environment import TicTacToeEnv


# Default hyperparameters (γ=1.0 recommended for finite episodic 3×3)
DEFAULT_HP = dict(
    alpha=0.1,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_end=0.05,
)


def _hp(overrides: dict | None = None) -> dict:
    h = dict(DEFAULT_HP)
    if overrides:
        h.update(overrides)
    return h


def _stats_str(values: list, as_percent: bool = True) -> str:
    if len(values) <= 1:
        if not values:
            return "N/A"
        return f"{values[0]:.2%}" if as_percent else f"{values[0]:.0f}"
    m = np.mean(values)
    s = np.std(values)
    return f"{m:.2%} ± {s:.2%}" if as_percent else f"{m:.0f} ± {s:.0f}"


def _print_run_summary(label: str, values: dict):
    print(f"\n  {label}")
    print(f"  {'-' * 60}")
    for metric, vals in values.items():
        print(f"    {metric:30s}: {_stats_str(vals, as_percent=(metric != 'q_table_size'))}")


def experiment_1(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    hp: dict,
    use_shaping: bool,
    alternate_first: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Exploration Schedule Ablation (vs Random Opponent)")
    print(f"  γ={hp['gamma']}  shaping={use_shaping}  |  Runs: {num_runs}  seed: {base_seed}")
    print("=" * 70)

    schedules = ["fixed", "linear", "exponential"]
    all_histories: dict = {}
    all_summaries: dict = {}
    opt_scores: dict = {}

    for sched in schedules:
        print(f"\n--- Schedule: {sched} ---")

        run_metrics = {
            "win_vs_random": [],
            "draw_vs_random": [],
            "loss_vs_random": [],
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
            "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                schedule=sched,
                total_episodes=total_episodes,
                **hp,
            )

            hist = train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
                use_shaping=use_shaping,
                algorithm="td",
                alternate_first=alternate_first,
            )

            if run == num_runs - 1:
                all_histories[f"ε-{sched}"] = hist
                agent.save(f"results/agent_exp1_{sched}.pkl")

            label = f"ε-{sched} (vs Random)"
            summary = print_final_summary(label, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"ε-{sched} (vs Random) — {num_runs} runs", run_metrics)
        opt_scores[f"ε-{sched}"] = np.mean(run_metrics["policy_optimality"])

        all_summaries[f"ε-{sched}"] = {
            "num_runs": num_runs,
            "base_seed": base_seed,
            "gamma": hp["gamma"],
            "use_shaping": use_shaping,
            "seeds": [base_seed + r for r in range(num_runs)],
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results", filename="exp1_schedules.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp1_optimality.png")

    with open("results/exp1_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n[Exp 1] Done. Figures saved to results/")


def experiment_2(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    hp: dict,
    alternate_first: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Training Paradigm – vs Random vs Self-Play")
    print(f"  γ={hp['gamma']}  |  Runs: {num_runs}  seed: {base_seed}")
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
            "win_vs_random": [],
            "draw_vs_random": [],
            "loss_vs_random": [],
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
            "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + 100 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            if mode == "train_vs_random":
                agent = QLearningAgent(
                    player=1, schedule="linear", total_episodes=total_episodes, **hp
                )
                hist = train_vs_random(
                    agent,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval,
                    verbose=False,
                    seed=seed,
                    alternate_first=alternate_first,
                )
                label = "vs Random"
            else:
                agent_sp1 = QLearningAgent(
                    player=1, schedule="linear", total_episodes=total_episodes, **hp
                )
                agent_sp2 = QLearningAgent(
                    player=2, schedule="linear", total_episodes=total_episodes, **hp
                )
                hist = train_selfplay(
                    agent_sp1,
                    agent_sp2,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval,
                    verbose=False,
                    seed=seed,
                )
                agent = agent_sp1
                label = "Self-play"

            if run == num_runs - 1:
                all_histories[paradigm] = hist
                if mode == "train_vs_random":
                    agent.save("results/agent_exp2_vs_random.pkl")
                else:
                    agent.save("results/agent_exp2_selfplay.pkl")

            summary = print_final_summary(f"{label} (run {run+1})", agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"{paradigm} — {num_runs} runs", run_metrics)
        opt_scores[paradigm] = np.mean(run_metrics["policy_optimality"])

        all_summaries[paradigm] = {
            "num_runs": num_runs,
            "base_seed": base_seed,
            "gamma": hp["gamma"],
            "seeds": [base_seed + 100 + r for r in range(num_runs)],
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results", filename="exp2_paradigms.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp2_optimality.png")

    with open("results/exp2_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n[Exp 2] Done. Figures saved to results/")


def experiment_3(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    hp: dict,
    alternate_first: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Policy Optimality vs Minimax (All Configurations)")
    print(f"  γ={hp['gamma']}  |  Runs: {num_runs}  seed: {base_seed}")
    print("=" * 70)

    configs = [
        ("vs_Random + fixed_ε", dict(schedule="fixed"), "fixed"),
        ("vs_Random + linear_ε", dict(schedule="linear"), "linear"),
        ("vs_Random + exp_ε", dict(schedule="exponential"), "exponential"),
    ]

    opt_scores: dict = {}
    all_results: dict = {}

    for idx, (label, extra, save_name) in enumerate(configs):
        print(f"\n  Training: {label}")

        run_metrics = {
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
        }

        for run in range(num_runs):
            seed = base_seed + 200 + idx * 10 + run
            print(f"    [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                total_episodes=total_episodes,
                **{**hp, **extra},
            )
            train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
                alternate_first=alternate_first,
            )

            if run == num_runs - 1:
                agent.save(f"results/agent_exp3_{save_name}.pkl")

            opt = compute_policy_optimality(agent, num_states=800)
            w, d, l = evaluate_vs_minimax(agent, episodes=300, agent_player=1)
            w2, d2, l2 = evaluate_vs_minimax(agent, episodes=300, agent_player=2)

            run_metrics["policy_optimality"].append(opt)
            run_metrics["win_vs_minimax"].append(w / 300)
            run_metrics["draw_vs_minimax"].append(d / 300)
            run_metrics["loss_vs_minimax"].append(l / 300)
            run_metrics["win_vs_minimax_p2"].append(w2 / 300)
            run_metrics["draw_vs_minimax_p2"].append(d2 / 300)
            run_metrics["loss_vs_minimax_p2"].append(l2 / 300)

            print(
                f"      Opt={opt:.2%}  "
                f"P1 W/D/L={w/300:.2%}/{d/300:.2%}/{l/300:.2%}  "
                f"P2 W/D/L={w2/300:.2%}/{d2/300:.2%}/{l2/300:.2%}"
            )

        opt_scores[label] = np.mean(run_metrics["policy_optimality"])
        all_results[label] = {
            "num_runs": num_runs,
            "gamma": hp["gamma"],
            "seeds": [base_seed + 200 + idx * 10 + r for r in range(num_runs)],
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }
        _print_run_summary(label, run_metrics)

    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp3_optimality_all.png")

    with open("results/exp3_minimax_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Exp 3] Done. Figures saved to results/")


def experiment_4_gamma_ablation(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    alternate_first: bool = False,
) -> None:
    """Compare γ=0.9 vs γ=1.0 under linear ε, vs random."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Discount Factor γ Ablation (linear ε, vs Random)")
    print(f"  Runs: {num_runs}  seed: {base_seed}")
    print("=" * 70)

    gammas = [0.9, 1.0]
    all_results: dict = {}
    opt_scores: dict = {}

    for gi, gamma in enumerate(gammas):
        label = f"γ={gamma:g}"
        print(f"\n--- {label} ---")
        hp = _hp({"gamma": gamma})

        run_metrics = {
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
            "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + 400 + gi * 20 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1,
                schedule="linear",
                total_episodes=total_episodes,
                **hp,
            )
            train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
                alternate_first=alternate_first,
            )

            if run == num_runs - 1:
                agent.save(f"results/agent_exp4_gamma_{str(gamma).replace('.', 'p')}.pkl")

            summary = print_final_summary(label, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"{label} — {num_runs} runs", run_metrics)
        opt_scores[label] = np.mean(run_metrics["policy_optimality"])
        all_results[label] = {
            "gamma": gamma,
            "num_runs": num_runs,
            "seeds": [base_seed + 400 + gi * 20 + r for r in range(num_runs)],
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }

    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp4_gamma_optimality.png")

    with open("results/exp4_gamma_summaries.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Exp 4] Done. -> results/exp4_gamma_summaries.json")


def experiment_5_curriculum(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    hp: dict,
    transition_fraction: float,
    mixed_prob: float,
    use_shaping: bool,
    alternate_first: bool = False,
) -> None:
    """vs Random, random→selfplay, mixed random/selfplay."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Curriculum / Mixed Opponents (linear ε)")
    print(
        f"  γ={hp['gamma']}  transition={transition_fraction}  mix_p={mixed_prob}  "
        f"shaping={use_shaping}"
    )
    print("=" * 70)

    configs = [
        ("vs_random", "rand"),
        ("random_then_selfplay", "rts"),
        ("mixed_selfplay", "mix"),
    ]
    all_histories: dict = {}
    all_results: dict = {}
    opt_scores: dict = {}

    for ci, (name, tag) in enumerate(configs):
        print(f"\n--- {name} ---")
        run_metrics = {
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
            "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + 500 + ci * 30 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1, schedule="linear", total_episodes=total_episodes, **hp
            )

            if name == "vs_random":
                hist = train_vs_random(
                    agent,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval,
                    verbose=False,
                    seed=seed,
                    use_shaping=use_shaping,
                    alternate_first=alternate_first,
                )
            elif name == "random_then_selfplay":
                hist = train_vs_random_then_selfplay(
                    agent,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval,
                    verbose=False,
                    seed=seed,
                    transition_fraction=transition_fraction,
                    use_shaping=use_shaping,
                    alternate_first=alternate_first,
                )
            else:
                agent2 = QLearningAgent(
                    player=2, schedule="linear", total_episodes=total_episodes, **hp
                )
                hist = train_mixed_random_selfplay(
                    agent,
                    agent2,
                    selfplay_prob=mixed_prob,
                    total_episodes=total_episodes,
                    eval_interval=eval_interval,
                    verbose=False,
                    seed=seed,
                    use_shaping=use_shaping,
                    alternate_first=alternate_first,
                )

            if run == num_runs - 1:
                all_histories[name] = hist
                agent.save(f"results/agent_exp5_{tag}.pkl")

            summary = print_final_summary(name, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"{name} — {num_runs} runs", run_metrics)
        opt_scores[name] = np.mean(run_metrics["policy_optimality"])
        all_results[name] = {
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }

    plot_training_curves(all_histories, save_dir="results", filename="exp5_curriculum.png")
    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp5_curriculum_opt.png")

    with open("results/exp5_curriculum_summaries.json", "w") as f:
        json.dump(
            {
                "hyperparameters": {**hp, "transition_fraction": transition_fraction, "mixed_prob": mixed_prob},
                "use_shaping": use_shaping,
                "configs": all_results,
            },
            f,
            indent=2,
        )

    print("\n[Exp 5] Done. -> results/exp5_curriculum_summaries.json")


def experiment_6_td_vs_mc(
    total_episodes: int,
    eval_interval: int,
    num_runs: int,
    base_seed: int,
    hp: dict,
    alternate_first: bool = False,
) -> None:
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: TD(0) vs Monte Carlo (linear ε, vs Random)")
    print(f"  γ={hp['gamma']}  |  Runs: {num_runs}")
    print("=" * 70)

    modes = [("TD", "td"), ("MC", "mc")]
    all_results: dict = {}
    opt_scores: dict = {}

    for mi, (label, algo) in enumerate(modes):
        print(f"\n--- {label} ---")
        run_metrics = {
            "win_vs_minimax": [],
            "draw_vs_minimax": [],
            "loss_vs_minimax": [],
            "win_vs_minimax_p2": [],
            "draw_vs_minimax_p2": [],
            "loss_vs_minimax_p2": [],
            "policy_optimality": [],
            "q_table_size": [],
        }

        for run in range(num_runs):
            seed = base_seed + 600 + mi * 25 + run
            print(f"  [Run {run+1}/{num_runs}] seed={seed}")

            agent = QLearningAgent(
                player=1, schedule="linear", total_episodes=total_episodes, **hp
            )
            train_vs_random(
                agent,
                total_episodes=total_episodes,
                eval_interval=eval_interval,
                verbose=False,
                seed=seed,
                algorithm=algo,
                alternate_first=alternate_first,
            )

            if run == num_runs - 1:
                agent.save(f"results/agent_exp6_{algo}.pkl")

            summary = print_final_summary(label, agent, verbose=False)
            for k in run_metrics:
                run_metrics[k].append(summary[k])

        _print_run_summary(f"{label} — {num_runs} runs", run_metrics)
        opt_scores[label] = np.mean(run_metrics["policy_optimality"])
        all_results[label] = {
            "algorithm": algo,
            "mean": {k: np.mean(v) for k, v in run_metrics.items()},
            "std": {k: np.std(v) for k, v in run_metrics.items()},
            "individual": [{k: run_metrics[k][i] for k in run_metrics} for i in range(num_runs)],
        }

    plot_policy_optimality_bar(opt_scores, save_dir="results", filename="exp6_td_vs_mc.png")

    with open("results/exp6_td_vs_mc_summaries.json", "w") as f:
        json.dump({"hyperparameters": hp, "results": all_results}, f, indent=2)

    print("\n[Exp 6] Done. -> results/exp6_td_vs_mc_summaries.json")


def play_vs_agent(agent_path: str = "results/agent_exp2_vs_random.pkl") -> None:
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


def parse_args():
    p = argparse.ArgumentParser(description="Q-Learning Tic-Tac-Toe")
    p.add_argument("--episodes", type=int, default=100_000, help="Total training episodes")
    p.add_argument("--eval-interval", type=int, default=2_000, help="Eval checkpoint interval")
    p.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Run only experiment 1–6 (default: all 1–3 legacy; use explicit list via none=all legacy)",
    )
    p.add_argument("--num-runs", type=int, default=3, help="Runs per config")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 1.0 from DEFAULT_HP)",
    )
    p.add_argument(
        "--potential-shaping",
        action="store_true",
        help="Use potential-based shaping in Exp 1 / 5 (vs-random parts)",
    )
    p.add_argument(
        "--transition-fraction",
        type=float,
        default=0.5,
        help="Exp5 random→selfplay: fraction of episodes in phase 1",
    )
    p.add_argument(
        "--mixed-prob",
        type=float,
        default=0.5,
        help="Exp5 mixed: probability of self-play episode",
    )
    p.add_argument(
        "--include-extended",
        action="store_true",
        help="When no --experiment, also run experiments 4–6 after 1–3",
    )
    p.add_argument(
        "--alternate-first",
        action="store_true",
        help="Strictly alternate agent as X vs O each episode (vs-random phases); "
        "default is random X/O per episode",
    )
    p.add_argument("--play", action="store_true", help="Interactive play vs saved agent")
    p.add_argument("--agent-path", type=str, default="results/agent_exp2_vs_random.pkl")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    if args.play:
        play_vs_agent(args.agent_path)
        return

    hp = _hp({"gamma": args.gamma} if args.gamma is not None else None)
    ep = args.episodes
    iv = args.eval_interval
    nr = args.num_runs
    seed = args.seed
    use_shaping = args.potential_shaping
    af = args.alternate_first

    if args.experiment is None:
        experiment_1(ep, iv, nr, seed, hp, use_shaping=use_shaping, alternate_first=af)
        experiment_2(ep, iv, nr, seed, hp, alternate_first=af)
        experiment_3(ep, iv, nr, seed, hp, alternate_first=af)
        if args.include_extended:
            experiment_4_gamma_ablation(ep, iv, nr, seed, alternate_first=af)
            experiment_5_curriculum(
                ep,
                iv,
                nr,
                seed,
                hp,
                args.transition_fraction,
                args.mixed_prob,
                use_shaping,
                alternate_first=af,
            )
            experiment_6_td_vs_mc(ep, iv, nr, seed, hp, alternate_first=af)
            print("\n[Done] Experiments 1–6 complete. Results in results/*.json and *.png.")
        else:
            print("\n[Done] Experiments 1–3 complete. Pass --include-extended to also run 4–6.")
        return

    exp = args.experiment
    if exp == 1:
        experiment_1(ep, iv, nr, seed, hp, use_shaping=use_shaping, alternate_first=af)
    elif exp == 2:
        experiment_2(ep, iv, nr, seed, hp, alternate_first=af)
    elif exp == 3:
        experiment_3(ep, iv, nr, seed, hp, alternate_first=af)
    elif exp == 4:
        experiment_4_gamma_ablation(ep, iv, nr, seed, alternate_first=af)
    elif exp == 5:
        experiment_5_curriculum(
            ep,
            iv,
            nr,
            seed,
            hp,
            args.transition_fraction,
            args.mixed_prob,
            use_shaping,
            alternate_first=af,
        )
    elif exp == 6:
        experiment_6_td_vs_mc(ep, iv, nr, seed, hp, alternate_first=af)


if __name__ == "__main__":
    main()
