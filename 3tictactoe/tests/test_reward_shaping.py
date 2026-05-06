"""Small tests for potential-based shaping helpers."""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.reward_shaping import potential_phi, shaped_reward


class TestRewardShaping(unittest.TestCase):
    def test_phi_empty_board(self) -> None:
        s = (0,) * 9
        self.assertEqual(potential_phi(s), 0.0)

    def test_phi_one_x_corner(self) -> None:
        s = [0] * 9
        s[0] = 1
        self.assertGreaterEqual(potential_phi(tuple(s)), 0.0)

    def test_shaped_intermediate_zero_r(self) -> None:
        s0 = (0,) * 9
        s1 = [0] * 9
        s1[0] = 1
        s1 = tuple(s1)
        gamma = 1.0
        r_tilde = shaped_reward(0.0, s0, s1, gamma)
        self.assertAlmostEqual(r_tilde, gamma * potential_phi(s1) - potential_phi(s0))


if __name__ == "__main__":
    unittest.main()
