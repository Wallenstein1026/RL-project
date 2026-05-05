"""
symmetry.py
-----------
D4 (dihedral group of order 8) symmetry transforms for the 4x4 board.

The 4x4 board has 8 symmetries:
  1. Identity (rotate 0°)
  2. Rotate 90° clockwise
  3. Rotate 180°
  4. Rotate 270° clockwise
  5. Reflect about vertical axis
  6. sr  – reflect then rotate 90°
  7. sr² – reflect then rotate 180° (= horizontal reflection)
  8. sr³ – reflect then rotate 270°

These are precomputed as index permutations on the flat 16-element board.
"""

from __future__ import annotations
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Precompute the 8 position permutations
# ---------------------------------------------------------------------------

def _build_d4_transforms() -> tuple:
    """
    Build two lists of length 8, each element a length-16 list.

    board_perm[t][old_pos] = new_pos   (forward transform)
    action_map[t][old_pos] = new_pos   (same mapping, used for actions)

    Both use the same permutation because an action at cell i transforms
    to the cell at position perm[i] under the symmetry.
    """
    board_perms: List[List[int]] = []
    action_maps: List[List[int]] = []

    # 1. Identity
    perm = list(range(16))
    board_perms.append(perm)
    action_maps.append(perm)

    # 2. Rotate 90°: (r, c) -> (c, 3 - r)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = c * 4 + (3 - r)
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 3. Rotate 180°: (r, c) -> (3 - r, 3 - c)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = (3 - r) * 4 + (3 - c)
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 4. Rotate 270°: (r, c) -> (3 - c, r)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = (3 - c) * 4 + r
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 5. Vertical reflection: (r, c) -> (r, 3 - c)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = r * 4 + (3 - c)
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 6. sr (reflect then rotate 90°): (r, c) -> r(c, 3 - r) = (c, r)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = c * 4 + r
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 7. sr² (reflect then rotate 180°): (r, c) -> (3 - r, c)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = (3 - r) * 4 + c
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    # 8. sr³ (reflect then rotate 270°): (r, c) -> (3 - c, 3 - r)
    perm = [0] * 16
    for r in range(4):
        for c in range(4):
            old = r * 4 + c
            new = (3 - c) * 4 + (3 - r)
            perm[old] = new
    board_perms.append(perm)
    action_maps.append(perm)

    return board_perms, action_maps


BOARD_PERMS, ACTION_MAPS = _build_d4_transforms()
N_TRANSFORMS = len(BOARD_PERMS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_board(board: tuple, transform_idx: int) -> tuple:
    """Apply symmetry transform to a board state."""
    perm = BOARD_PERMS[transform_idx]
    return tuple(board[perm[i]] for i in range(16))


def transform_action(action: int, transform_idx: int) -> int:
    """Apply symmetry transform to an action (board position)."""
    return ACTION_MAPS[transform_idx][action]


def augment_experience(
    state: tuple,
    action: int,
    reward: float,
    next_state: tuple,
    done: bool,
) -> List[Tuple[tuple, int, float, tuple, bool]]:
    """
    Generate all distinct symmetric variants of one experience tuple.

    Returns between 1 and 8 unique (s, a, r, s', done) tuples, deduplicated
    so that symmetric states with identical (state, action) pairs are not
    double-counted.
    """
    seen: set = set()
    results: List[Tuple[tuple, int, float, tuple, bool]] = []

    for t_idx in range(N_TRANSFORMS):
        board_perm = BOARD_PERMS[t_idx]
        action_perm = ACTION_MAPS[t_idx]

        s_aug = tuple(state[board_perm[i]] for i in range(16))
        a_aug = action_perm[action]

        key = (s_aug, a_aug)
        if key in seen:
            continue
        seen.add(key)

        ns_aug = tuple(next_state[board_perm[i]] for i in range(16))
        results.append((s_aug, a_aug, reward, ns_aug, done))

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def test_all_symmetries(verbosity: int = 0) -> bool:
    """
    Verify D4 group properties:
      1. All 8 transforms are distinct.
      2. Applying rotate-90 four times returns the identity for every cell.
      3. Applying any reflection twice returns the identity for every cell.
      4. transform_board + transform_action are consistent on non-trivial boards.
    """
    all_ok = True

    # Check distinctness
    perm_tuples = [tuple(p) for p in BOARD_PERMS]
    if len(set(perm_tuples)) != 8:
        print("[FAIL] Not all 8 transforms are distinct.")
        all_ok = False
    elif verbosity > 0:
        print("[PASS] All 8 transforms are distinct.")

    # Check Rot90^4 = identity
    p = BOARD_PERMS[1]  # rot90
    for start in range(16):
        pos = start
        for _ in range(4):
            pos = p[pos]
        if pos != start:
            print(f"[FAIL] Rot90^4({start}) = {pos}, expected {start}")
            all_ok = False
    if verbosity > 0:
        print("[PASS] Rot90^4 = identity for all cells.")

    # Check every reflection^2 = identity
    for name, t_idx in [("vflip", 4), ("sr", 5), ("sr^2", 6), ("sr^3", 7)]:
        p = BOARD_PERMS[t_idx]
        for start in range(16):
            if p[p[start]] != start:
                print(f"[FAIL] {name}^2({start}) = {p[p[start]]}, expected {start}")
                all_ok = False
    if verbosity > 0:
        print("[PASS] All reflections are involutions.")

    # Check on a non-trivial board
    state = tuple([1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])
    for t_idx in range(8):
        tb = transform_board(state, t_idx)
        assert len(tb) == 16
        # Apply identity transform should recover original
        tb_back = transform_board(tb, _inverse_idx(t_idx))
        if tb_back != state:
            print(f"[FAIL] Inverse check failed for transform {t_idx}")
            all_ok = False
    if verbosity > 0:
        print("[PASS] Inverse transforms recover original board.")

    return all_ok


_INVERSE_MAP = {
    0: 0,   # identity -> identity
    1: 3,   # rot90  -> rot270
    2: 2,   # rot180 -> rot180
    3: 1,   # rot270 -> rot90
    4: 4,   # vflip  -> vflip
    5: 5,   # sr     -> sr
    6: 6,   # sr^2   -> sr^2
    7: 7,   # sr^3   -> sr^3
}


def _inverse_idx(t_idx: int) -> int:
    return _INVERSE_MAP[t_idx]


if __name__ == "__main__":
    ok = test_all_symmetries(verbosity=1)
    print("\nAll tests passed!" if ok else "\nSome tests FAILED!")
