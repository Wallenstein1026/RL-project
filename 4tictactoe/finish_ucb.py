"""
Finish UCB experiment — run the vs_RandomThenSelfPlay that crashed,
then Part 3 evaluation against Minimax.

Part 1 (c-grid, vs_Random) and Part 2 (vs_Random, vs_SelfPlay) already
completed in the original run. Only vs_RandomThenSelfPlay + Part 3 remain.
"""
import json
import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from src.agents import UCBQLearningAgent
from src.training import train_vs_random_then_selfplay
from src.evaluation import (
    print_final_summary,
    evaluate_vs_minimax,
    compute_policy_optimality,
)

TOTAL_EPISODES = 500_000
EVAL_INTERVAL = 2_000
BEST_C = 0.5  # from Part 1: highest draw_vs_minimax mean = 0.22

# ========================================================================== #
#  1. vs_RandomThenSelfPlay (3 seeds)
# ========================================================================== #
print("=" * 70)
print(f"  UCB FINISH: vs_RandomThenSelfPlay (c={BEST_C})")
print("=" * 70)

seeds = [742, 743, 744]
label = "vs_RandomThenSelfPlay + ucb"

run_metrics = {
    "win_vs_random": [], "draw_vs_random": [], "loss_vs_random": [],
    "win_vs_minimax": [], "draw_vs_minimax": [], "loss_vs_minimax": [],
    "policy_optimality": [], "q_table_size": [],
}

def _run_score(metrics):
    return (
        metrics.get("draw_vs_minimax", 0.0),
        metrics.get("policy_optimality", 0.0),
        metrics.get("win_vs_random", 0.0),
        -metrics.get("loss_vs_minimax", 0.0),
    )

best_run = None

for run in range(len(seeds)):
    seed = seeds[run]
    print(f"  [Run {run+1}/{len(seeds)}] seed={seed}")

    agent = UCBQLearningAgent(player=1, c=BEST_C)
    hist = train_vs_random_then_selfplay(
        agent, total_episodes=TOTAL_EPISODES,
        eval_interval=EVAL_INTERVAL, verbose=False, seed=seed)

    summary = print_final_summary(label, agent, verbose=False)
    for k in run_metrics:
        run_metrics[k].append(summary[k])

    score = _run_score(summary)
    if best_run is None or score > best_run["score"]:
        best_run = {
            "agent": agent, "metrics": summary, "run_index": run,
            "seed": seed, "score": score, "history": hist,
        }

    print(f"      Opt={summary['policy_optimality']:.2%}  "
          f"D_M={summary['draw_vs_minimax']:.2%}  "
          f"Q={summary['q_table_size']}")

best_run["agent"].save("results/agent_ucb2_vs_RandomThenSelfPlay.pkl")

rts_result = {
    "paradigm": "vs_RandomThenSelfPlay",
    "c": BEST_C,
    "total_episodes": TOTAL_EPISODES,
    "eval_interval": EVAL_INTERVAL,
    "num_runs": len(seeds),
    "seeds": seeds,
    "hyperparameters": {"alpha": 0.1, "gamma": 0.9, "c": BEST_C},
    "best_run": {
        "run_index": best_run["run_index"],
        "seed": best_run["seed"],
        "selection_score": list(best_run["score"]),
        "metrics": best_run["metrics"],
    },
    "mean": {k: float(np.mean(v)) for k, v in run_metrics.items()},
    "std":  {k: float(np.std(v)) for k, v in run_metrics.items()},
    "individual": [{k: run_metrics[k][i] for k in run_metrics}
                    for i in range(len(seeds))],
}

print(f"\n  {label} ({len(seeds)} runs)")
print(f"  {'-' * 60}")
for k, v in run_metrics.items():
    is_pct = k != "q_table_size"
    if is_pct:
        print(f"    {k:30s}: {np.mean(v):.2%} ± {np.std(v):.2%}")
    else:
        print(f"    {k:30s}: {np.mean(v):.0f} ± {np.std(v):.0f}")

# ========================================================================== #
#  2. Save complete exp_ucb2_summaries.json
# ========================================================================== #
# We only have per-seed data for vs_RandomThenSelfPlay.
# For vs_Random and vs_SelfPlay, use saved best agents for evaluation.
ucb2_results = {}

for paradigm, pkl_file in [
    ("vs_Random", "results/agent_ucb2_vs_Random.pkl"),
    ("vs_SelfPlay", "results/agent_ucb2_vs_SelfPlay.pkl"),
]:
    label_p2 = f"{paradigm} + ucb"
    agent_p2 = UCBQLearningAgent(player=1, c=BEST_C)
    agent_p2.load(pkl_file)
    opt = compute_policy_optimality(agent_p2, num_states=800)
    w, d, l = evaluate_vs_minimax(agent_p2, episodes=300)
    ucb2_results[label_p2] = {
        "paradigm": paradigm,
        "c": BEST_C,
        "note": "Best-agent eval only (per-seed lost due to crash before JSON save)",
        "best_agent_metrics": {
            "policy_optimality": opt,
            "draw_vs_minimax": d / 300,
            "win_vs_minimax": w / 300,
            "loss_vs_minimax": l / 300,
        },
    }

ucb2_results[f"vs_RandomThenSelfPlay + ucb"] = rts_result

with open("results/exp_ucb2_summaries.json", "w") as f:
    json.dump(ucb2_results, f, indent=2)
print("\n  exp_ucb2_summaries.json saved")

# ========================================================================== #
#  3. Part 3: vs Minimax Baseline (evaluate Part 1 agents)
# ========================================================================== #
print("\n" + "=" * 70)
print("  UCB PART 3: vs Minimax Baseline")
print("=" * 70)

c_grid = [0.5, 1.0, 2.0, 5.0]
all_results_ucb3 = {}

for c_val in c_grid:
    label_key = f"ucb_c_{c_val:g}"
    display_label = f"vs_Random + ucb_c_{c_val:g}"
    c_str = f"{c_val:g}".replace(".", "p")
    pkl_path = f"results/agent_ucb1_{c_str}.pkl"

    print(f"\n  Evaluating {display_label}")
    agent = UCBQLearningAgent(player=1, c=c_val)
    agent.load(pkl_path)

    opt = compute_policy_optimality(agent, num_states=800)
    w, d, l = evaluate_vs_minimax(agent, episodes=300)
    summary = {
        "win_vs_minimax": w / 300,
        "draw_vs_minimax": d / 300,
        "loss_vs_minimax": l / 300,
        "policy_optimality": opt,
    }
    print(f"      Opt={opt:.2%}  W={w/300:.2%}  D={d/300:.2%}  L={l/300:.2%}")

    all_results_ucb3[label_key] = {
        "c": c_val,
        "hyperparameters": {"alpha": 0.1, "gamma": 0.9, "c": c_val},
        "metrics": summary,
    }

with open("results/exp_ucb3_summaries.json", "w") as f:
    json.dump(all_results_ucb3, f, indent=2)
print(f"\n  exp_ucb3_summaries.json saved")

print("\n[Done] UCB experiment complete!")
print("Files:")
print("  results/exp_ucb1_summaries.json  (Part 1 - already complete)")
print("  results/exp_ucb2_summaries.json  (Part 2)")
print("  results/exp_ucb3_summaries.json  (Part 3)")
