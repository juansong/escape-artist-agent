# tests/test_env.py
import numpy as np
from escape_artist.envs.escape_env import EscapeEnv, EnvConfig, TRAP, GOAL

def make_env(**over):
    cfg = EnvConfig(
        size=(6, 6),
        start=(0, 0),
        goal=(5, 5),
        traps_pct=0.08,
        r_safe=1,
        slip=0.0,
        lethal_traps=True,
        max_steps=30,
        layout_mode="per_env",
        seed=123,
        obs_mode="pos",
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return EscapeEnv(cfg)

def test_reset_and_bounds():
    env = make_env()
    obs, info = env.reset()
    assert env.action_space.n == 4
    x, y = info["pos"]
    assert 0 <= x < env.w and 0 <= y < env.h

def test_slip_changes_action_effect():
    env = make_env(slip=1.0)  # always slip to random neighbor
    env.reset()
    x0, y0 = env.agent_pos
    # Take 'right' multiple times; due to slip, position should not always be (x0+1,y0)
    moved_diff = False
    for _ in range(10):
        _, _, _, _, info = env.step(1)  # intended "right"
        if info["pos"] != (min(x0 + 1, env.w - 1), y0):
            moved_diff = True
            break
    assert moved_diff  # slip actually changed effect

def test_truncation_and_termination():
    env = make_env(max_steps=5)
    env.reset()
    terminated = truncated = False
    for _ in range(5):
        _, r, term, trunc, _ = env.step(1)
        terminated |= term
        truncated = trunc
        if term or trunc:
            break
    assert truncated or terminated  # at least one ending condition triggers

def test_layout_id_stable_per_env_mode():
    env = make_env(layout_mode="per_env", seed=7)
    _, info1 = env.reset()
    id1 = info1["layout_id"]
    # multiple resets keep the same layout in per_env mode
    for _ in range(3):
        _, info2 = env.reset()
        assert info2["layout_id"] == id1
