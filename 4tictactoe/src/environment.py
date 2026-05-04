"""
environment.py
--------------
4x4 Tic-Tac-Toe environment modelled as a Markov Decision Process (MDP).

State  : tuple of 16 integers (0 = empty, 1 = X, 2 = O)
Action : integer 0-15 (board position)
Reward : +1 win, -1 loss, +0.5 draw, 0 otherwise

The win rule is connect 4 in any row, column, or diagonal.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


BOARD_SIZE = 4
WIN_LENGTH = 4
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE


def generate_win_lines(board_size: int = BOARD_SIZE,
                       win_length: int = WIN_LENGTH) -> List[Tuple[int, ...]]:
    """Generate all contiguous winning lines for a square board."""
    directions = [
        (0, 1),    # horizontal
        (1, 0),    # vertical
        (1, 1),    # main diagonal
        (1, -1),   # anti diagonal
    ]
    lines: List[Tuple[int, ...]] = []

    for row in range(board_size):
        for col in range(board_size):
            for dr, dc in directions:
                end_row = row + (win_length - 1) * dr
                end_col = col + (win_length - 1) * dc
                if 0 <= end_row < board_size and 0 <= end_col < board_size:
                    line = tuple(
                        (row + i * dr) * board_size + (col + i * dc)
                        for i in range(win_length)
                    )
                    lines.append(line)

    return lines


WIN_LINES = generate_win_lines()


class TicTacToeEnv:
    """
    Two-player 4x4 Tic-Tac-Toe environment.

    Players are identified by integers:
        1: X (the learning agent, always moves first)
        2: O (the opponent)

    The board is stored as a flat tuple so it can be used directly as a
    dictionary key in the Q-table.
    """

    def __init__(self) -> None:
        self.board: List[int] = [0] * BOARD_CELLS
        self.current_player: int = 1
        self.done: bool = False
        self.winner: Optional[int] = None

    def reset(self) -> tuple:
        """Reset the board and return the initial state."""
        self.board = [0] * BOARD_CELLS
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
        Apply action for the current player.

        Returns (next_state, reward_for_player1, done, info).
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset() first.")
        if action < 0 or action >= BOARD_CELLS:
            raise ValueError(f"Action {action} is outside the board.")
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        self.board[action] = self.current_player
        reward, self.done, self.winner = self._check_terminal()

        if not self.done:
            self.current_player = 3 - self.current_player

        return self.get_state(), reward, self.done, {"winner": self.winner}

    def _check_terminal(self) -> Tuple[float, bool, Optional[int]]:
        """Evaluate the board after the current player's move."""
        p = self.current_player

        for line in WIN_LINES:
            if all(self.board[idx] == p for idx in line):
                reward = 1.0 if p == 1 else -1.0
                return reward, True, p

        if 0 not in self.board:
            return 0.5, True, 0

        return 0.0, False, None

    @staticmethod
    def check_winner(state: tuple) -> Optional[int]:
        """
        Check winner for an arbitrary board state.

        Returns 1, 2, 0 (draw), or None (game ongoing).
        """
        for line in WIN_LINES:
            values = [state[idx] for idx in line]
            if values[0] != 0 and all(v == values[0] for v in values):
                return values[0]
        if 0 not in state:
            return 0
        return None

    def render(self, state: Optional[tuple] = None) -> None:
        """Print a human-readable board."""
        if state is None:
            state = self.get_state()
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for row in range(BOARD_SIZE):
            rows.append(
                " ".join(
                    f"{symbols[state[row * BOARD_SIZE + col]]:>2s}"
                    for col in range(BOARD_SIZE)
                )
            )
        print("\n".join(rows))
        print()
