# escape_artist/envs/generators.py
from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np

EMPTY, TRAP, GOAL, WALL = 0, 1, 2, 3  # simple tile codes


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def exclusion_mask(
    size: Tuple[int, int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    r_safe: int,
) -> np.ndarray:
    """Boolean mask of cells that are NOT allowed to be traps."""
    h, w = size
    mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if (x, y) == start or (x, y) == goal:
                mask[y, x] = True
                continue
            if r_safe > 0 and (
                manhattan((x, y), start) <= r_safe or manhattan((x, y), goal) <= r_safe
            ):
                mask[y, x] = True
    return mask


def neighbors_4(x: int, y: int, w: int, h: int) -> Iterable[Tuple[int, int]]:
    if y > 0:
        yield (x, y - 1)
    if x < w - 1:
        yield (x + 1, y)
    if y < h - 1:
        yield (x, y + 1)
    if x > 0:
        yield (x - 1, y)


def path_exists_bfs(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS that treats TRAP and WALL as blocked, others as free."""
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    if (sx, sy) == (gx, gy):
        return True

    seen = np.zeros((h, w), dtype=bool)
    q: List[Tuple[int, int]] = [(sx, sy)]
    seen[sy, sx] = True

    while q:
        x, y = q.pop(0)
        for nx, ny in neighbors_4(x, y, w, h):
            if seen[ny, nx]:
                continue
            if grid[ny, nx] in (TRAP, WALL):
                continue
            if (nx, ny) == (gx, gy):
                return True
            seen[ny, nx] = True
            q.append((nx, ny))
    return False


def sample_trap_layout(
    rng: np.random.Generator,
    size: Tuple[int, int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    traps_pct: float,
    r_safe: int,
    max_resample: int = 20,
) -> np.ndarray:
    """
    Sample a random trap layout with a guaranteed path from start to goal.
    Uses binomial sampling of trap count and rejection sampling via BFS.

    Returns a grid with {EMPTY, TRAP, GOAL}.
    """
    h, w = size
    base_traps_pct = traps_pct
    excl = exclusion_mask(size, start, goal, r_safe)
    free_indices = np.argwhere(~excl)  # candidate cells for traps

    for _ in range(max_resample):
        grid = np.zeros((h, w), dtype=np.int8)
        # how many traps to place among candidates?
        k = rng.binomial(n=len(free_indices), p=traps_pct)
        if k > 0:
            sel_idx = rng.choice(len(free_indices), size=k, replace=False)
            for idx in sel_idx:
                y, x = free_indices[idx]
                grid[y, x] = TRAP
        gx, gy = goal
        grid[gy, gx] = GOAL

        if path_exists_bfs(grid, start, goal):
            return grid

        # relax density slightly and retry
        traps_pct = max(0.0, traps_pct * 0.9)

    # last resort: empty grid with goal
    grid = np.zeros((h, w), dtype=np.int8)
    gx, gy = goal
    grid[gy, gx] = GOAL
    return grid
