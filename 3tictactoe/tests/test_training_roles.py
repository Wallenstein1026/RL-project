"""Smoke tests for vs-random training with agent as X or O."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from src.agents import QLearningAgent, RandomAgent
from src.environment import TicTacToeEnv
from src.training import (
    _p1_reward_to_agent_value,
    _run_episode_vs_random_mc_roles,
    _run_episode_vs_random_roles,
    train_vs_random,
)


def test_p1_reward_to_agent_value():
    assert _p1_reward_to_agent_value(1.0, 1) == 1.0
    assert _p1_reward_to_agent_value(-1.0, 1) == -1.0
    assert _p1_reward_to_agent_value(0.5, 1) == 0.5
    assert _p1_reward_to_agent_value(1.0, 2) == -1.0
    assert _p1_reward_to_agent_value(-1.0, 2) == 1.0
    assert _p1_reward_to_agent_value(0.5, 2) == 0.5
    assert _p1_reward_to_agent_value(0.0, 2) == 0.0


def test_run_episode_roles_agent_as_o_updates_q():
    np.random.seed(0)
    env = TicTacToeEnv()
    agent = QLearningAgent(
        player=2, schedule="fixed", total_episodes=100, epsilon_start=0.3, epsilon_end=0.3
    )
    opp = RandomAgent(player=1)
    n_before = len(agent.q_table)
    for _ in range(30):
        _run_episode_vs_random_roles(env, agent, opp, agent_player=2, use_shaping=False)
    assert len(agent.q_table) > n_before
    assert agent.player == 2


def test_run_episode_mc_roles_agent_as_o():
    np.random.seed(1)
    env = TicTacToeEnv()
    agent = QLearningAgent(
        player=2, schedule="fixed", total_episodes=100, epsilon_start=0.5, epsilon_end=0.5
    )
    opp = RandomAgent(player=1)
    n_before = len(agent.q_table)
    for _ in range(40):
        _run_episode_vs_random_mc_roles(env, agent, opp, agent_player=2)
    assert len(agent.q_table) > n_before


def test_train_vs_random_short_no_crash():
    agent = QLearningAgent(
        player=1, schedule="fixed", total_episodes=50, epsilon_start=0.4, epsilon_end=0.4
    )
    train_vs_random(
        agent,
        total_episodes=20,
        eval_interval=10,
        eval_episodes=4,
        verbose=False,
        seed=123,
        alternate_first=True,
        algorithm="td",
    )
    train_vs_random(
        agent,
        total_episodes=10,
        eval_interval=10,
        eval_episodes=4,
        verbose=False,
        seed=124,
        alternate_first=False,
        algorithm="mc",
    )
