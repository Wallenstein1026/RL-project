"""
agents.py
---------
Agents used in training and evaluation:

  QLearningAgent  - tabular Q-learning with epsilon-greedy exploration
  RandomAgent     - selects a legal move uniformly at random
  MinimaxAgent    - depth-limited alpha-beta search baseline for 4x4 play
"""

from __future__ import annotations
import math
import pickle
from typing import Dict, List, Tuple

import numpy as np

from .environment import BOARD_SIZE, BOARD_CELLS, TicTacToeEnv, WIN_LINES


# =========================================================================== #
#  Exploration-schedule helpers                                                 #
# =========================================================================== #

def make_epsilon_schedule(schedule: str,
                          epsilon_start: float,
                          epsilon_end: float,
                          total_episodes: int):
    """
    Factory function that returns a callable episode -> epsilon.

    Parameters
    ----------
    schedule        : 'fixed' | 'linear' | 'exponential'
    epsilon_start   : initial epsilon value
    epsilon_end     : minimum epsilon value (used by linear/exponential)
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
        Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        player: int = 1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        schedule: str = "linear",
        total_episodes: int = 100_000,
        use_symmetry: bool = False,
    ) -> None:
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.schedule = schedule
        self.use_symmetry = use_symmetry

        self.q_table: Dict[Tuple[tuple, int], float] = {}
        self._eps_fn = make_epsilon_schedule(
            schedule, epsilon_start, epsilon_end, total_episodes
        )
        self.epsilon: float = epsilon_start

        self.random_action_count: int = 0
        self.greedy_action_count: int = 0
        self._sym_count: int = 0

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
            return int(np.random.choice(available_actions))

        self.greedy_action_count += 1
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return int(np.random.choice(best_actions))

    def _update_single(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: List[int],
        done: bool,
    ) -> float:
        """Single Q-update without symmetry augmentation."""
        current_q = self.get_q(state, action)

        if done or not next_available:
            target = reward
        else:
            target = reward + self.gamma * self.get_max_q(next_state, next_available)

        td_error = target - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        return td_error

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: List[int],
        done: bool,
    ) -> float:
        if self.use_symmetry:
            from .symmetry import augment_experience

            experiences = augment_experience(state, action, reward,
                                            next_state, done)
            total_td = 0.0
            for s_aug, a_aug, r_aug, ns_aug, d_aug in experiences:
                if d_aug:
                    ns_avail: List[int] = []
                else:
                    ns_avail = [i for i, v in enumerate(ns_aug) if v == 0]
                    if not ns_avail:
                        d_aug = True
                total_td += abs(self._update_single(s_aug, a_aug, r_aug,
                                                    ns_aug, ns_avail, d_aug))
            self._sym_count += len(experiences)
            return total_td / max(len(experiences), 1)

        return self._update_single(state, action, reward, next_state,
                                   next_available, done)

    def update_epsilon(self, episode: int) -> None:
        self.epsilon = self._eps_fn(episode)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[QLearningAgent] Q-table saved to {path} "
              f"({len(self.q_table)} entries)")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"[QLearningAgent] Q-table loaded from {path} "
              f"({len(self.q_table)} entries)")


# =========================================================================== #
#  UCB Q-Learning Agent                                                         #
# =========================================================================== #

class UCBQLearningAgent:
    """
    Tabular Q-learning with Upper Confidence Bound (UCB) exploration.

    Instead of epsilon-greedy, selects actions via:

        a* = argmax [ Q(s,a) + c * sqrt(ln(N_s + 1) / (N(s,a) + 1e-8)) ]

    where N_s = total visits to state s, N(s,a) = visits to (s,a).
    Never-tried actions (N(s,a) == 0) get infinite bonus → always tried first.
    """

    def __init__(
        self,
        player: int = 1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        c: float = 2.0,
        use_symmetry: bool = False,
    ) -> None:
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.use_symmetry = use_symmetry

        self.q_table: Dict[Tuple[tuple, int], float] = {}
        self._sa_counts: Dict[Tuple[tuple, int], int] = {}
        self._s_counts: Dict[tuple, int] = {}

        # For training log compatibility
        self.random_action_count: int = 0
        self.greedy_action_count: int = 0
        self._sym_count: int = 0
        self.epsilon: float = 0.0  # compatibility

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

        if greedy:
            q_values = [self.get_q(state, a) for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(available_actions, q_values)
                            if q == max_q]
            return int(np.random.choice(best_actions))

        n_s = self._s_counts.get(state, 0) + 1
        best_value = -float("inf")
        best_actions: List[int] = []
        max_q_only = -float("inf")
        best_q_actions: List[int] = []

        for a in available_actions:
            q = self.get_q(state, a)
            if q > max_q_only:
                max_q_only = q
                best_q_actions = [a]
            elif q == max_q_only:
                best_q_actions.append(a)

            n_sa = self._sa_counts.get((state, a), 0)
            if n_sa == 0:
                bonus = float("inf")
            else:
                bonus = self.c * math.sqrt(math.log(n_s) / n_sa)

            ucb = q + bonus
            if ucb > best_value:
                best_value = ucb
                best_actions = [a]
            elif ucb == best_value:
                best_actions.append(a)

        chosen = int(np.random.choice(best_actions))
        if chosen in best_q_actions:
            self.greedy_action_count += 1
        else:
            self.random_action_count += 1
        return chosen

    def _update_single(
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

        # Update visit counts
        self._sa_counts[(state, action)] = self._sa_counts.get((state, action), 0) + 1
        self._s_counts[state] = self._s_counts.get(state, 0) + 1

        return td_error

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_available: List[int],
        done: bool,
    ) -> float:
        if self.use_symmetry:
            from .symmetry import augment_experience

            experiences = augment_experience(state, action, reward,
                                            next_state, done)
            total_td = 0.0
            for s_aug, a_aug, r_aug, ns_aug, d_aug in experiences:
                if d_aug:
                    ns_avail: List[int] = []
                else:
                    ns_avail = [i for i, v in enumerate(ns_aug) if v == 0]
                    if not ns_avail:
                        d_aug = True
                total_td += abs(self._update_single(s_aug, a_aug, r_aug,
                                                    ns_aug, ns_avail, d_aug))
            self._sym_count += len(experiences)
            return total_td / max(len(experiences), 1)

        return self._update_single(state, action, reward, next_state,
                                   next_available, done)

    def update_epsilon(self, episode: int) -> None:
        """UCB doesn't use epsilon schedules. No-op for compatibility."""
        pass

    def save(self, path: str) -> None:
        data = {
            "q_table": self.q_table,
            "sa_counts": self._sa_counts,
            "s_counts": self._s_counts,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[UCBQLearningAgent] Saved to {path} "
              f"(Q: {len(self.q_table)}, SA-counts: {len(self._sa_counts)}, "
              f"S-counts: {len(self._s_counts)})")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self._sa_counts = data.get("sa_counts", {})
        self._s_counts = data.get("s_counts", {})
        print(f"[UCBQLearningAgent] Loaded from {path} "
              f"({len(self.q_table)} Q-entries)")


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
#  Depth-Limited Minimax Agent                                                  #
# =========================================================================== #

class MinimaxAgent:
    """
    Depth-limited alpha-beta search baseline for 4x4 connect-4 Tic-Tac-Toe.

    Full-width exact minimax is much more expensive on 4x4 than on 3x3, so this
    agent uses a configurable search depth plus a line-count heuristic. It keeps
    the same public API as the 3x3 baseline for evaluation compatibility.
    """

    def __init__(self, player: int = 2, depth_limit: int = 3) -> None:
        self.player = player
        self.depth_limit = depth_limit
        self._cache: Dict[Tuple[tuple, int, int, int], float] = {}

    def select_action(self, state: tuple, available_actions: List[int],
                      greedy: bool = True) -> int:
        best_score = -float("inf")
        best_action = available_actions[0]

        for action in self._ordered_actions(state, available_actions):
            score = self.score_action(state, action)
            if score > best_score:
                best_score, best_action = score, action

        return best_action

    def score_action(self, state: tuple, action: int) -> float:
        """
        Score taking action from state as this agent's player.

        Higher is better from this agent's perspective.
        """
        next_state = list(state)
        next_state[action] = self.player
        next_state_tuple = tuple(next_state)

        winner = TicTacToeEnv.check_winner(next_state_tuple)
        if winner is not None:
            return self._terminal_score(winner, self.player, self.depth_limit)

        return self._search(
            next_state_tuple,
            current_player=3 - self.player,
            depth=self.depth_limit - 1,
            alpha=-float("inf"),
            beta=float("inf"),
            root_player=self.player,
        )

    def best_score(self, state: tuple, available_actions: List[int]) -> float:
        """Return the best search score among the available actions."""
        return max(self.score_action(state, a) for a in available_actions)

    def _search(self, state: tuple, current_player: int, depth: int,
                alpha: float, beta: float, root_player: int) -> float:
        winner = TicTacToeEnv.check_winner(state)
        if winner is not None:
            return self._terminal_score(winner, root_player, depth)
        if depth <= 0:
            return self._heuristic(state, root_player)

        key = (state, current_player, depth, root_player)
        if key in self._cache:
            return self._cache[key]

        available = [i for i, v in enumerate(state) if v == 0]
        ordered = self._ordered_actions(state, available)

        if current_player == root_player:
            value = -float("inf")
            for action in ordered:
                ns = list(state)
                ns[action] = current_player
                value = max(
                    value,
                    self._search(tuple(ns), 3 - current_player,
                                 depth - 1, alpha, beta, root_player),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for action in ordered:
                ns = list(state)
                ns[action] = current_player
                value = min(
                    value,
                    self._search(tuple(ns), 3 - current_player,
                                 depth - 1, alpha, beta, root_player),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break

        self._cache[key] = value
        return value

    @staticmethod
    def _terminal_score(winner: int, root_player: int, depth: int) -> float:
        if winner == root_player:
            return 1_000_000 + depth
        if winner == 3 - root_player:
            return -1_000_000 - depth
        return 0.0

    @staticmethod
    def _heuristic(state: tuple, root_player: int) -> float:
        opponent = 3 - root_player
        weights = {1: 1.0, 2: 8.0, 3: 80.0, 4: 1_000_000.0}
        score = 0.0

        for line in WIN_LINES:
            values = [state[idx] for idx in line]
            root_count = values.count(root_player)
            opponent_count = values.count(opponent)

            if root_count and opponent_count:
                continue
            if root_count:
                score += weights[root_count]
            elif opponent_count:
                score -= 1.1 * weights[opponent_count]

        center_start = BOARD_SIZE // 2 - 1
        center_end = BOARD_SIZE // 2 + 1
        for row in range(center_start, center_end):
            for col in range(center_start, center_end):
                idx = row * BOARD_SIZE + col
                if state[idx] == root_player:
                    score += 0.5
                elif state[idx] == opponent:
                    score -= 0.5

        return score

    @staticmethod
    def _ordered_actions(state: tuple, available_actions: List[int]) -> List[int]:
        center = (BOARD_SIZE - 1) / 2
        return sorted(
            available_actions,
            key=lambda action: (
                abs(action // BOARD_SIZE - center) +
                abs(action % BOARD_SIZE - center)
            ),
        )
