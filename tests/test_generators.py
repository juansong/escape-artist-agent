# tests/test_generators.py
import numpy as np
from escape_artist.envs.generators import (
    EMPTY, TRAP, GOAL,
    sample_trap_layout, exclusion_mask, path_exists_bfs,
)

def test_exclusion_zone_and_goal():
    rng = np.random.default_rng(0)
    size = (8, 8)
    start = (0, 0)
    goal = (7, 7)
    r_safe = 2
    grid = sample_trap_layout(rng, size, start, goal, traps_pct=0.12, r_safe=r_safe, max_resample=20)

    # start/goal never traps
    sx, sy = start
    gx, gy = goal
    assert grid[sy, sx] != TRAP
    assert grid[gy, gx] == GOAL

    # safe radius respected (no traps in exclusion mask)
    mask = exclusion_mask(size, start, goal, r_safe)
    ys, xs = np.where(mask)
    assert not np.any(grid[ys, xs] == TRAP)

def test_solvability():
    rng = np.random.default_rng(123)
    size = (10, 10)
    start = (0, 0)
    goal = (9, 9)
    grid = sample_trap_layout(rng, size, start, goal, traps_pct=0.1, r_safe=1, max_resample=20)
    assert path_exists_bfs(grid, start, goal)

def test_density_within_tolerance():
    rng = np.random.default_rng(777)
    size = (12, 12)
    start, goal = (0, 0), (11, 11)
    traps_pct = 0.15
    # average over multiple samples to smooth rejection sampling effects
    ks = []
    for _ in range(20):
        grid = sample_trap_layout(rng, size, start, goal, traps_pct=traps_pct, r_safe=1, max_resample=5)
        ks.append(np.mean(grid == TRAP))
    mean_density = float(np.mean(ks))
    assert 0.07 <= mean_density <= 0.20  # loose bounds (rejection may reduce density)

def test_deterministic_with_seed_and_episode_index_hash():
    # sampling determinism in the helper itself
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    size = (6, 6)
    start, goal = (0, 0), (5, 5)
    g1 = sample_trap_layout(rng1, size, start, goal, traps_pct=0.1, r_safe=1, max_resample=10)
    g2 = sample_trap_layout(rng2, size, start, goal, traps_pct=0.1, r_safe=1, max_resample=10)
    assert np.array_equal(g1, g2)
