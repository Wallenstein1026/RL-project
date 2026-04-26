"""
training.py
-----------
Training routines for Q-learning agents in Tic-Tac-Toe.

Two paradigms are supported:
  • train_vs_random  – agent (player 1) vs a fixed random opponent (player 2)
  • train_selfplay   – two Q-learning agents play against each other
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from .environment import TicTacToeEnv
from .agents import QLearningAgent, RandomAgent


# =========================================================================== #
#  Shared episode helpers                                                       #
# =========================================================================== #

def _run_episode_vs_random(
    env: TicTacToeEnv,
    agent: QLearningAgent,
    opponent: RandomAgent,
) -> float:
    state = env.reset()
    total_reward = 0.0

    while True:
        available = env.get_available_actions(state)
        action = agent.select_action(state, available)

        next_state, reward, done, _ = env.step(action)
        if done:
            agent.update(state, action, reward, next_state, [], True)
            total_reward += reward
            break

        opp_available = env.get_available_actions(next_state)
        opp_action = opponent.select_action(next_state, opp_available)

        next_next_state, opp_reward, done, _ = env.step(opp_action)
        if done:
            agent.update(state, action, opp_reward, next_next_state, [], True)
            total_reward += opp_reward
            break

        next_available = env.get_available_actions(next_next_state)
        agent.update(state, action, 0.0, next_next_state,
                     next_available, False)

        state = next_next_state

    return total_reward


def _run_episode_selfplay(
    env: TicTacToeEnv,
    agent1: QLearningAgent,
    agent2: QLearningAgent,
) -> Tuple[float, float]:
    state = env.reset()
    r1 = r2 = 0.0

    last_exp: Dict[int, Optional[Tuple[tuple, int]]] = {1: None, 2: None}

    while True:
        current_player = env.current_player
        agent = agent1 if current_player == 1 else agent2

        available = env.get_available_actions(state)
        action = agent.select_action(state, available)

        next_state, reward, done, _ = env.step(action)

        if current_player == 1:
            agent_reward = reward
            r1 += agent_reward
        else:
            agent_reward = -reward if reward != 0 else 0.0
            r2 += agent_reward

        other_player = 3 - current_player
        if last_exp[other_player] is not None:
            ls, la = last_exp[other_player]
            other_agent = agent2 if other_player == 2 else agent1
            if done:
                other_agent.update(ls, la, -agent_reward,
                                   next_state, [], True)
            else:
                other_agent.update(ls, la, 0.0,
                                   next_state,
                                   env.get_available_actions(next_state),
                                   False)

        if done:
            agent.update(state, action, agent_reward, next_state, [], True)
            break

        last_exp[current_player] = (state, action)
        state = next_state

    return r1, r2


# =========================================================================== #
#  Main training functions                                                      #
# =========================================================================== #

def train_vs_random(
    agent: QLearningAgent,
    total_episodes: int = 100_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 500,
    verbose: bool = True,
) -> Dict:
    env = TicTacToeEnv()
    opponent = RandomAgent(player=2)

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
    }

    reward_buffer: List[float] = []

    for ep in range(total_episodes):
        agent.update_epsilon(ep)
        reward = _run_episode_vs_random(env, agent, opponent)
        reward_buffer.append(reward)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random
            wins, draws, _ = evaluate_vs_random(agent, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))

            total_actions = agent.random_action_count + agent.greedy_action_count
            rand_frac = (agent.random_action_count / total_actions
                         if total_actions > 0 else 0.0)

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)

            if verbose:
                print(
                    f"[Ep {ep+1:>7d}]  ε={agent.epsilon:.4f}  "
                    f"avg_r={avg_r:+.3f}  "
                    f"win={wins/eval_episodes:.2%}  "
                    f"draw={draws/eval_episodes:.2%}"
                )

            agent.random_action_count = 0
            agent.greedy_action_count = 0

    return history


def train_selfplay(
    agent1: QLearningAgent,
    agent2: QLearningAgent,
    total_episodes: int = 100_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 500,
    verbose: bool = True,
) -> Dict:
    env = TicTacToeEnv()

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
    }

    reward_buffer: List[float] = []

    for ep in range(total_episodes):
        agent1.update_epsilon(ep)
        agent2.update_epsilon(ep)

        r1, _ = _run_episode_selfplay(env, agent1, agent2)
        reward_buffer.append(r1)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random
            wins, draws, _ = evaluate_vs_random(agent1, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))

            total_actions = agent1.random_action_count + agent1.greedy_action_count
            rand_frac = (agent1.random_action_count / total_actions
                         if total_actions > 0 else 0.0)

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent1.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)

            if verbose:
                print(
                    f"[Ep {ep+1:>7d}]  ε={agent1.epsilon:.4f}  "
                    f"avg_r={avg_r:+.3f}  "
                    f"win={wins/eval_episodes:.2%}  "
                    f"draw={draws/eval_episodes:.2%}"
                )

            agent1.random_action_count = 0
            agent1.greedy_action_count = 0

    return history
