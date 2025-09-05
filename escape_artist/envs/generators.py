# escape_artist/envs/generators.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from collections import deque

EMPTY = 0
TRAP  = 1
GOAL  = 2

def exclusion_mask(size: Tuple[int,int], start: Tuple[int,int], goal: Tuple[int,int], r_safe: int) -> np.ndarray:
    """Return boolean mask True where traps are NOT allowed (square/Chebyshev radius)."""
    H, W = size
    sx, sy = start
    gx, gy = goal
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((np.abs(xx - sx) <= r_safe) & (np.abs(yy - sy) <= r_safe)) \
         | ((np.abs(xx - gx) <= r_safe) & (np.abs(yy - gy) <= r_safe))
    # Always protect start & goal
    mask[sy, sx] = True
    mask[gy, gx] = True
    return mask

def path_exists_bfs(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> bool:
    """Check 4-neighbor path exists from start to goal avoiding TRAP cells."""
    H, W = grid.shape
    sx, sy = start
    gx, gy = goal
    if grid[sy, sx] == TRAP:
        return False
    if grid[gy, gx] == TRAP:
        return False
    q = deque()
    q.append((sx, sy))
    visited = np.zeros_like(grid, dtype=bool)
    visited[sy, sx] = True
    dirs = [(0,-1),(1,0),(0,1),(-1,0)]
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for dx, dy in dirs:
            xn, yn = x + dx, y + dy
            if 0 <= xn < W and 0 <= yn < H and not visited[yn, xn] and grid[yn, xn] != TRAP:
                visited[yn, xn] = True
                q.append((xn, yn))
    return False

def sample_trap_layout(rng: np.random.Generator,
                       size: Tuple[int,int],
                       start: Tuple[int,int],
                       goal: Tuple[int,int],
                       traps_pct: float,
                       r_safe: int,
                       max_resample: int = 50) -> np.ndarray:
    """
    Sample a grid with traps according to density, enforcing exclusion around start/goal,
    and resample until a path exists between start and goal (up to max_resample).
    Returns grid with values {EMPTY, TRAP, GOAL}.
    """
    H, W = size
    sx, sy = start
    gx, gy = goal
    excl = exclusion_mask(size, start, goal, r_safe)

    for _ in range(max_resample):
        grid = np.zeros((H, W), dtype=np.int8)
        # sample traps everywhere except exclusion mask and goal
        candidates = np.ones((H, W), dtype=bool)
        candidates[excl] = False
        trap_mask = rng.random((H, W)) < traps_pct
        trap_mask &= candidates
        grid[trap_mask] = TRAP

        # ensure start clear, set goal
        grid[sy, sx] = EMPTY
        grid[gy, gx] = GOAL

        # enforce solvability
        if path_exists_bfs(grid, start, goal):
            return grid

    # last resort: return trap-sparse grid with only goal if repeated failures
    grid = np.zeros((H, W), dtype=np.int8)
    grid[gy, gx] = GOAL
    return grid
