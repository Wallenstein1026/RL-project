"""
agents.py
---------
Agents used in training and evaluation:

  QLearningAgent  – tabular Q-learning with ε-greedy exploration
  RandomAgent     – selects a legal move uniformly at random
  MinimaxAgent    – plays the game-theoretically optimal strategy (used only
                    during evaluation; never trained)
"""

from __future__ import annotations
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from .environment import TicTacToeEnv, WIN_LINES


# =========================================================================== #
#  Exploration-schedule helpers                                                 #
# =========================================================================== #

def make_epsilon_schedule(schedule: str,
                          epsilon_start: float,
                          epsilon_end: float,
                          total_episodes: int):
    """
    Factory function that returns a callable  episode → ε.

    Parameters
    ----------
    schedule        : 'fixed' | 'linear' | 'exponential'
    epsilon_start   : initial ε value
    epsilon_end     : minimum ε value (used by linear/exponential)
    total_episodes  : total number of training episodes
    """
    if schedule == "fixed":
        def eps_fn(episode: int) -> float:
            return epsilon_start

    elif schedule == "linear":
        def eps_fn(episode: int) -> float:
            fraction = min(episode / total_episodes, 1.0)
            return epsilon_start + fraction * (epsilon_end - epsilon_start)

    elif schedule == "exponential":
        decay = (epsilon_end / epsilon_start) ** (1.0 / total_episodes)

        def eps_fn(episode: int) -> float:
            return max(epsilon_start * (decay ** episode), epsilon_end)

    else:
        raise ValueError(f"Unknown epsilon schedule: '{schedule}'")

    return eps_fn


# =========================================================================== #
#  Q-Learning Agent                                                             #
# =========================================================================== #

class QLearningAgent:
    """
    Tabular Q-learning agent (off-policy TD control).

    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_{a'} Q(s',a') − Q(s,a)]

    Parameters
    ----------
    player          : 1 or 2 (which mark the agent places)
    alpha           : learning rate
    gamma           : discount factor
    epsilon_start   : initial exploration rate
    epsilon_end     : minimum exploration rate
    schedule        : 'fixed' | 'linear' | 'exponential'
    total_episodes  : total episodes (used by decay schedules)
    """

    def __init__(
        self,
        player: int = 1,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        schedule: str = "linear",
        total_episodes: int = 100_000,
    ) -> None:
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.schedule = schedule

        self.q_table: Dict[Tuple[tuple, int], float] = {}
        self._eps_fn = make_epsilon_schedule(
            schedule, epsilon_start, epsilon_end, total_episodes
        )
        self.epsilon: float = epsilon_start

        self.random_action_count: int = 0
        self.greedy_action_count: int = 0

    def get_q(self, state: tuple, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def get_max_q(self, state: tuple, available_actions: List[int]) -> float:
        if not available_actions:
            return 0.0
        return max(self.get_q(state, a) for a in available_actions)

    def select_action(self, state: tuple, available_actions: List[int],
                      greedy: bool = False) -> int:
        if not available_actions:
            raise ValueError("No available actions.")

        if not greedy and np.random.random() < self.epsilon:
            self.random_action_count += 1
            return np.random.choice(available_actions)

        self.greedy_action_count += 1
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return np.random.choice(best_actions)

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: List[int],
        done: bool,
    ) -> float:
        current_q = self.get_q(state, action)

        if done or not next_available:
            target = reward
        else:
            target = reward + self.gamma * self.get_max_q(next_state, next_available)

        td_error = target - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        return td_error

    def update_mc(self, state: tuple, action: int, g_return: float) -> float:
        """Every-visit MC / full-return target: Q ← Q + α (G − Q)."""
        current_q = self.get_q(state, action)
        td_error = g_return - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        return td_error

    def update_epsilon(self, episode: int) -> None:
        self.epsilon = self._eps_fn(episode)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[QLearningAgent] Q-table saved to {path}  "
              f"({len(self.q_table)} entries)")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"[QLearningAgent] Q-table loaded from {path}  "
              f"({len(self.q_table)} entries)")


# =========================================================================== #
#  Random Agent                                                                 #
# =========================================================================== #

class RandomAgent:
    """Selects a legal action uniformly at random."""

    def __init__(self, player: int = 2) -> None:
        self.player = player

    def select_action(self, state: tuple, available_actions: List[int],
                      greedy: bool = False) -> int:
        return int(np.random.choice(available_actions))


# =========================================================================== #
#  Minimax Agent (evaluation baseline only)                                    #
# =========================================================================== #

class MinimaxAgent:
    """
    Game-theoretically optimal agent via the Minimax algorithm.

    Used exclusively as a fixed evaluation baseline — never trained.
    Caches computed scores to speed up repeated queries.
    """

    def __init__(self, player: int = 2) -> None:
        self.player = player
        self._cache: Dict[Tuple[tuple, int], int] = {}

    def select_action(self, state: tuple, available_actions: List[int],
                      greedy: bool = True) -> int:
        # score_action returns from current player's perspective, so
        # both players maximize their own score.
        best_score = -float("inf")
        best_action = available_actions[0]

        for action in available_actions:
            score = self.score_action(state, action)
            if score > best_score:
                best_score, best_action = score, action

        return best_action

    def score_action(self, state: tuple, action: int) -> int:
        """
        Compute the minimax score for taking *action* from *state*
        as the current player.

        Returns +1 (win), 0 (draw), -1 (loss) from current player's perspective.
        """
        next_state = list(state)
        next_state[action] = self.player
        next_state = tuple(next_state)
        score = self._minimax(next_state, player=3 - self.player,
                              is_maximizing=(self.player == 2))
        # Convert score back to current-player perspective:
        # _minimax returns +1 for X(1) win, -1 for O(2) win, 0 draw.
        # If we are player 1, score is already from our perspective.
        # If we are player 2, we need to invert.
        if self.player == 2:
            score = -score
        return score

    def best_score(self, state: tuple, available_actions: List[int]) -> int:
        """
        Return the best possible minimax score achievable from *state*
        for the current player, among the given available actions.
        Since score_action returns from current player's perspective,
        both players maximize their own score.
        """
        return max(self.score_action(state, a) for a in available_actions)

    def _minimax(self, state: tuple, player: int,
                 is_maximizing: bool) -> int:
        key = (state, player)
        if key in self._cache:
            return self._cache[key]

        winner = TicTacToeEnv.check_winner(state)
        if winner == 1:
            return 1
        if winner == 2:
            return -1
        if winner == 0:
            return 0

        available = [i for i, v in enumerate(state) if v == 0]

        if is_maximizing:
            best = -2
            for action in available:
                ns = list(state); ns[action] = player; ns = tuple(ns)
                best = max(best, self._minimax(ns, 3 - player, False))
        else:
            best = 2
            for action in available:
                ns = list(state); ns[action] = player; ns = tuple(ns)
                best = min(best, self._minimax(ns, 3 - player, True))

        self._cache[key] = best
        return best
