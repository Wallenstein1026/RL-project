"""
environment.py
--------------
Tic-Tac-Toe game environment modelled as a Markov Decision Process (MDP).

State  : tuple of 9 integers  (0 = empty, 1 = X, 2 = O)
Action : integer 0-8 (board position)
Reward : +1 win, -1 loss, +0.5 draw, 0 otherwise
"""

from __future__ import annotations
from typing import Optional, Tuple, List


# --------------------------------------------------------------------------- #
#  Win conditions                                                               #
# --------------------------------------------------------------------------- #
WIN_LINES: List[Tuple[int, int, int]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),               # diagonals
]


class TicTacToeEnv:
    """
    Two-player Tic-Tac-Toe environment.

    Players are identified by integers:
        1  →  X  (the learning agent, always moves first)
        2  →  O  (the opponent)

    The board is stored as a flat tuple of 9 cells so it can be used
    directly as a dictionary key in the Q-table.
    """

    def __init__(self) -> None:
        self.board: List[int] = [0] * 9
        self.current_player: int = 1          # player 1 always starts
        self.done: bool = False
        self.winner: Optional[int] = None     # None | 1 | 2 | 0 (draw)

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple:
        """Reset the board and return the initial state."""
        self.board = [0] * 9
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self) -> tuple:
        """Return the current board as a hashable tuple."""
        return tuple(self.board)

    def get_available_actions(self, state: Optional[tuple] = None) -> List[int]:
        """Return a list of empty cell indices."""
        if state is None:
            state = self.get_state()
        return [i for i, v in enumerate(state) if v == 0]

    def step(self, action: int) -> Tuple[tuple, float, bool, dict]:
        """
        Apply *action* for the current player.

        Returns
        -------
        next_state : tuple
        reward     : float   (from the perspective of player 1)
        done       : bool
        info       : dict    {'winner': 0|1|2|None}
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset() first.")
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        self.board[action] = self.current_player
        reward, self.done, self.winner = self._check_terminal()

        if not self.done:
            self.current_player = 3 - self.current_player  # 1↔2

        return self.get_state(), reward, self.done, {"winner": self.winner}

    # ------------------------------------------------------------------ #
    #  Terminal check                                                       #
    # ------------------------------------------------------------------ #

    def _check_terminal(self) -> Tuple[float, bool, Optional[int]]:
        """
        Evaluate the board after the current player's move.

        Returns (reward_for_player1, done, winner).
        """
        p = self.current_player

        for a, b, c in WIN_LINES:
            if self.board[a] == self.board[b] == self.board[c] == p:
                reward = 1.0 if p == 1 else -1.0
                return reward, True, p

        if 0 not in self.board:
            return 0.5, True, 0

        return 0.0, False, None

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_winner(state: tuple) -> Optional[int]:
        """
        Check winner for an arbitrary board state (static helper).

        Returns 1, 2, 0 (draw), or None (game ongoing).
        """
        for a, b, c in WIN_LINES:
            if state[a] == state[b] == state[c] != 0:
                return state[a]
        if 0 not in state:
            return 0
        return None

    def render(self, state: Optional[tuple] = None) -> None:
        """Print a human-readable board."""
        if state is None:
            state = self.get_state()
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for row in range(3):
            rows.append(" ".join(symbols[state[row * 3 + col]] for col in range(3)))
        print("\n".join(rows))
        print()
''