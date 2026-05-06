"""
training.py
-----------
Training routines for Q-learning agents in Tic-Tac-Toe.

Paradigms:
  • train_vs_random              – agent vs random; each episode agent is X or O at random
  • train_selfplay               – two Q-learning agents
  • train_vs_random_then_selfplay – first half vs random, then self-play
  • train_mixed_random_selfplay   – each episode: random or self-play with prob p
  • train_vs_random(..., algorithm="mc") – episodic Monte Carlo (same role sampling)
"""

from __future__ import annotations

import copy
import numpy as np
from typing import Dict, List, Optional, Tuple

from .environment import TicTacToeEnv
from .agents import QLearningAgent, RandomAgent


def _p1_reward_to_agent_value(r_p1: float, agent_player: int) -> float:
    """
    Map env step reward (always from player 1 / X perspective) to the learning
    agent's utility for that transition.
    """
    if agent_player == 1:
        return r_p1
    if r_p1 == 0.5:
        return 0.5
    if r_p1 == 0.0:
        return 0.0
    return -r_p1


def _maybe_shape(
    r: float,
    s_before: tuple,
    s_after: tuple,
    gamma: float,
    use_shaping: bool,
) -> float:
    if not use_shaping:
        return r
    from .reward_shaping import shaped_reward

    return shaped_reward(r, s_before, s_after, gamma)


def _run_episode_vs_random_roles(
    env: TicTacToeEnv,
    agent: QLearningAgent,
    opponent: RandomAgent,
    agent_player: int,
    use_shaping: bool = False,
) -> Tuple[float, float]:
    """
    One vs-random episode where the learning agent plays as ``agent_player`` (1 or 2).
    Opponent is RandomAgent(3 - agent_player). If the env opens on the opponent,
    the opponent moves first.
    """
    agent.player = agent_player
    opponent.player = 3 - agent_player

    state = env.reset()
    td_errors: List[float] = []
    gam = agent.gamma
    total_agent_reward = 0.0

    if env.current_player != agent_player:
        avail = env.get_available_actions(state)
        oa = opponent.select_action(state, avail)
        next_s, r_p1, done, _ = env.step(oa)
        state = next_s
        if done:
            return 0.0, 0.0

    while True:
        avail = env.get_available_actions(state)
        action = agent.select_action(state, avail)
        next_state, r_p1, done, _ = env.step(action)
        agent_r = _p1_reward_to_agent_value(r_p1, agent_player)
        if done:
            r_t = _maybe_shape(agent_r, state, next_state, gam, use_shaping)
            td_errors.append(abs(agent.update(state, action, r_t, next_state, [], True)))
            total_agent_reward += agent_r
            break

        oa = opponent.select_action(next_state, env.get_available_actions(next_state))
        next_next, r2_p1, done2, _ = env.step(oa)
        agent_r2 = _p1_reward_to_agent_value(r2_p1, agent_player)
        if done2:
            r_t = _maybe_shape(agent_r2, state, next_next, gam, use_shaping)
            td_errors.append(
                abs(agent.update(state, action, r_t, next_next, [], True))
            )
            total_agent_reward += agent_r2
            break

        n_avail = env.get_available_actions(next_next)
        r_t = _maybe_shape(0.0, state, next_next, gam, use_shaping)
        td_errors.append(
            abs(agent.update(state, action, r_t, next_next, n_avail, False))
        )
        state = next_next

    avg_td = float(np.mean(td_errors)) if td_errors else 0.0
    return total_agent_reward, avg_td


def _run_episode_vs_random_mc_roles(
    env: TicTacToeEnv,
    agent: QLearningAgent,
    opponent: RandomAgent,
    agent_player: int,
) -> Tuple[float, float]:
    """Monte Carlo on full trajectory; backward utility uses agent-scaled step rewards."""
    agent.player = agent_player
    opponent.player = 3 - agent_player

    transitions: List[Tuple[int, tuple, int, float]] = []
    state = env.reset()
    total_agent_reward = 0.0
    g = agent.gamma

    if env.current_player != agent_player:
        avail = env.get_available_actions(state)
        oa = opponent.select_action(state, avail)
        s_opp = state
        next_s, r_p1, done, _ = env.step(oa)
        transitions.append((3 - agent_player, s_opp, oa, r_p1))
        total_agent_reward += _p1_reward_to_agent_value(r_p1, agent_player)
        state = next_s
        if done:
            return total_agent_reward, 0.0

    while True:
        avail = env.get_available_actions(state)
        action = agent.select_action(state, avail)
        next_s, r_p1, done, _ = env.step(action)
        transitions.append((agent_player, state, action, r_p1))
        total_agent_reward += _p1_reward_to_agent_value(r_p1, agent_player)
        if done:
            break
        state = next_s

        oa = opponent.select_action(state, env.get_available_actions(state))
        next_s2, r2, done2, _ = env.step(oa)
        transitions.append((3 - agent_player, state, oa, r2))
        total_agent_reward += _p1_reward_to_agent_value(r2, agent_player)
        if done2:
            break
        state = next_s2

    mc_errors: List[float] = []
    g_ret = 0.0
    for pl, s, a, r_p1 in reversed(transitions):
        ra = _p1_reward_to_agent_value(r_p1, agent_player)
        g_ret = ra + g * g_ret
        if pl == agent_player:
            mc_errors.append(abs(agent.update_mc(s, a, g_ret)))

    avg_td = float(np.mean(mc_errors)) if mc_errors else 0.0
    return total_agent_reward, avg_td


def _run_episode_selfplay(
    env: TicTacToeEnv,
    agent1: QLearningAgent,
    agent2: QLearningAgent,
) -> Tuple[float, float, float, float]:
    state = env.reset()
    r1 = r2 = 0.0
    td1: List[float] = []
    td2: List[float] = []

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
                td = other_agent.update(ls, la, -agent_reward, next_state, [], True)
            else:
                td = other_agent.update(
                    ls,
                    la,
                    0.0,
                    next_state,
                    env.get_available_actions(next_state),
                    False,
                )
            if other_player == 1:
                td1.append(abs(td))
            else:
                td2.append(abs(td))

        if done:
            td = agent.update(state, action, agent_reward, next_state, [], True)
            if current_player == 1:
                td1.append(abs(td))
            else:
                td2.append(abs(td))
            break

        last_exp[current_player] = (state, action)
        state = next_state

    avg_td1 = float(np.mean(td1)) if td1 else 0.0
    avg_td2 = float(np.mean(td2)) if td2 else 0.0
    return r1, r2, avg_td1, avg_td2


def train_vs_random(
    agent: QLearningAgent,
    total_episodes: int = 100_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 500,
    verbose: bool = True,
    seed: Optional[int] = None,
    use_shaping: bool = False,
    algorithm: str = "td",
    alternate_first: bool = False,
) -> Dict:
    if seed is not None:
        np.random.seed(seed)

    env = TicTacToeEnv()
    opponent = RandomAgent(player=2)

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
        "td_error": [],
    }

    reward_buffer: List[float] = []
    td_buffer: List[float] = []

    for ep in range(total_episodes):
        agent.update_epsilon(ep)
        ap = (1 + ep % 2) if alternate_first else int(np.random.randint(1, 3))
        if algorithm == "mc":
            reward, avg_td = _run_episode_vs_random_mc_roles(
                env, agent, opponent, ap
            )
        else:
            reward, avg_td = _run_episode_vs_random_roles(
                env, agent, opponent, ap, use_shaping=use_shaping
            )
        reward_buffer.append(reward)
        td_buffer.append(avg_td)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random

            wins, draws, _ = evaluate_vs_random(agent, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))
            avg_td_window = float(np.mean(td_buffer[-eval_interval:]))

            total_actions = agent.random_action_count + agent.greedy_action_count
            rand_frac = (
                agent.random_action_count / total_actions if total_actions > 0 else 0.0
            )

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)
            history["td_error"].append(avg_td_window)

            if verbose:
                tag = "MC" if algorithm == "mc" else "TD"
                print(
                    f"[Ep {ep+1:>7d}] [{tag}] ε={agent.epsilon:.4f}  "
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
    seed: Optional[int] = None,
) -> Dict:
    if seed is not None:
        np.random.seed(seed)

    env = TicTacToeEnv()

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
        "td_error": [],
    }

    reward_buffer: List[float] = []
    td_buffer: List[float] = []

    for ep in range(total_episodes):
        agent1.update_epsilon(ep)
        agent2.update_epsilon(ep)

        r1, _, td1, _ = _run_episode_selfplay(env, agent1, agent2)
        reward_buffer.append(r1)
        td_buffer.append(td1)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random

            wins, draws, _ = evaluate_vs_random(agent1, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))
            avg_td_window = float(np.mean(td_buffer[-eval_interval:]))

            total_actions = agent1.random_action_count + agent1.greedy_action_count
            rand_frac = (
                agent1.random_action_count / total_actions if total_actions > 0 else 0.0
            )

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent1.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)
            history["td_error"].append(avg_td_window)

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


def train_vs_random_then_selfplay(
    agent: QLearningAgent,
    total_episodes: int = 100_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 500,
    verbose: bool = True,
    seed: Optional[int] = None,
    transition_fraction: float = 0.5,
    use_shaping: bool = False,
    alternate_first: bool = False,
) -> Dict:
    """
    Phase 1: vs random. Phase 2: self-play vs a deepcopy of the agent at switch.
    transition_fraction in (0,1): fraction of episodes spent in phase 1.
    """
    if seed is not None:
        np.random.seed(seed)

    env = TicTacToeEnv()
    opponent_random = RandomAgent(player=2)
    transition_ep = int(total_episodes * transition_fraction)
    transition_ep = max(1, min(transition_ep, total_episodes - 1))

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
        "td_error": [],
    }

    reward_buffer: List[float] = []
    td_buffer: List[float] = []
    opponent_agent: Optional[QLearningAgent] = None

    for ep in range(total_episodes):
        agent.update_epsilon(ep)

        if ep < transition_ep:
            ap = (1 + ep % 2) if alternate_first else int(np.random.randint(1, 3))
            reward, avg_td = _run_episode_vs_random_roles(
                env, agent, opponent_random, ap, use_shaping=use_shaping
            )
            reward_buffer.append(reward)
            td_buffer.append(avg_td)
        else:
            if opponent_agent is None:
                opponent_agent = copy.deepcopy(agent)
                opponent_agent.player = 2
                if verbose:
                    print(f"[Ep {ep+1:>7d}] === Switching to self-play ===")

            opponent_agent.update_epsilon(ep)
            r1, _, td1, _ = _run_episode_selfplay(env, agent, opponent_agent)
            reward_buffer.append(r1)
            td_buffer.append(td1)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random

            wins, draws, _ = evaluate_vs_random(agent, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))
            avg_td_window = float(np.mean(td_buffer[-eval_interval:]))

            total_actions = agent.random_action_count + agent.greedy_action_count
            rand_frac = (
                agent.random_action_count / total_actions if total_actions > 0 else 0.0
            )

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)
            history["td_error"].append(avg_td_window)

            if verbose:
                phase = "SP" if ep >= transition_ep else "R"
                print(
                    f"[Ep {ep+1:>7d}] [{phase}] ε={agent.epsilon:.4f}  "
                    f"avg_r={avg_r:+.3f}  "
                    f"win={wins/eval_episodes:.2%}  "
                    f"draw={draws/eval_episodes:.2%}"
                )

            agent.random_action_count = 0
            agent.greedy_action_count = 0

    return history


def train_mixed_random_selfplay(
    agent: QLearningAgent,
    agent2: QLearningAgent,
    selfplay_prob: float = 0.5,
    total_episodes: int = 100_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 500,
    verbose: bool = True,
    seed: Optional[int] = None,
    use_shaping: bool = False,
    alternate_first: bool = False,
) -> Dict:
    """
    Each episode: with probability selfplay_prob run one self-play game
    (agent as P1, agent2 as P2); otherwise train agent vs random.
    """
    if seed is not None:
        np.random.seed(seed)

    env = TicTacToeEnv()
    opponent_random = RandomAgent(player=2)

    history = {
        "episode": [],
        "epsilon": [],
        "avg_reward": [],
        "win_rate_vs_random": [],
        "draw_rate_vs_random": [],
        "random_action_fraction": [],
        "td_error": [],
    }

    reward_buffer: List[float] = []
    td_buffer: List[float] = []

    for ep in range(total_episodes):
        agent.update_epsilon(ep)
        agent2.update_epsilon(ep)

        if np.random.random() < selfplay_prob:
            r1, _, td1, _ = _run_episode_selfplay(env, agent, agent2)
            reward_buffer.append(r1)
            td_buffer.append(td1)
        else:
            ap = (1 + ep % 2) if alternate_first else int(np.random.randint(1, 3))
            reward, avg_td = _run_episode_vs_random_roles(
                env, agent, opponent_random, ap, use_shaping=use_shaping
            )
            reward_buffer.append(reward)
            td_buffer.append(avg_td)

        if (ep + 1) % eval_interval == 0:
            from .evaluation import evaluate_vs_random

            wins, draws, _ = evaluate_vs_random(agent, episodes=eval_episodes)
            avg_r = float(np.mean(reward_buffer[-eval_interval:]))
            avg_td_window = float(np.mean(td_buffer[-eval_interval:]))

            total_actions = agent.random_action_count + agent.greedy_action_count
            rand_frac = (
                agent.random_action_count / total_actions if total_actions > 0 else 0.0
            )

            history["episode"].append(ep + 1)
            history["epsilon"].append(agent.epsilon)
            history["avg_reward"].append(avg_r)
            history["win_rate_vs_random"].append(wins / eval_episodes)
            history["draw_rate_vs_random"].append(draws / eval_episodes)
            history["random_action_fraction"].append(rand_frac)
            history["td_error"].append(avg_td_window)

            if verbose:
                print(
                    f"[Ep {ep+1:>7d}] [mix p={selfplay_prob:.2f}] ε={agent.epsilon:.4f}  "
                    f"avg_r={avg_r:+.3f}  "
                    f"win={wins/eval_episodes:.2%}  "
                    f"draw={draws/eval_episodes:.2%}"
                )

            agent.random_action_count = 0
            agent.greedy_action_count = 0

    return history
