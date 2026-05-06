"""
Potential-based reward shaping (Ng et al., 1999) for 3×3 tic-tac-toe.

Uses Φ(s) based on open two-in-a-row threats so that
    r' = r + γ Φ(s') − Φ(s)
preserves optimal policies under the usual assumptions when the MDP
discount matches γ.
"""

from __future__ import annotations

from typing import Tuple

from .environment import WIN_LINES

_THREAT_WEIGHT = 0.02


def potential_phi(state: tuple) -> float:
    """
    Heuristic potential: favor boards where X (1) has more open two-in-a-row
    threats than O (2). Bounded so terminal rewards still dominate.
    """
    val = 0.0
    for a, b, c in WIN_LINES:
        cells = [state[a], state[b], state[c]]
        if cells.count(1) == 2 and cells.count(0) == 1:
            val += _THREAT_WEIGHT
        if cells.count(2) == 2 and cells.count(0) == 1:
            val -= _THREAT_WEIGHT
    return val


def shaped_reward(
    r: float,
    s_before: tuple,
    s_after: tuple,
    gamma: float,
) -> float:
    """Potential-based shaping increment added to the MDP reward r."""
    return r + gamma * potential_phi(s_after) - potential_phi(s_before)
