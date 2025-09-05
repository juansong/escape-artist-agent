# tests/test_mc.py
import numpy as np
from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.algos.mc_control import MCConfig, train_mc_control, greedy_policy_rollout

def make_easy_env():
    cfg = EnvConfig(
        size=(6, 6),
        start=(0, 0),
        goal=(5, 5),
        traps_pct=0.05,
        r_safe=2,
        slip=0.0,
        lethal_traps=True,
        max_steps=80,
        layout_mode="per_env",
        seed=999,
        obs_mode="pos",
    )
    return EscapeEnv(cfg)

def test_mc_first_visit_reaches_goal_easy():
    env = make_easy_env()
    mc_cfg = MCConfig(
        episodes=3000, gamma=0.99, visit="first",
        epsilon_start=0.2, epsilon_end=0.05, epsilon_decay_episodes=2500,
        seed=123,
    )
    Q, returns = train_mc_control(env, mc_cfg)
    # a greedy rollout should finish with positive return (goal=+1, small step costs)
    total, traj = greedy_policy_rollout(env, Q, max_steps=env.cfg.max_steps)
    assert total > 0.5
    assert 1 < len(traj) <= env.cfg.max_steps

def test_mc_every_visit_not_worse_than_first():
    env = make_easy_env()
    cfg1 = MCConfig(episodes=2000, visit="first", seed=1)
    cfg2 = MCConfig(episodes=2000, visit="every", seed=1)
    Q1, r1 = train_mc_control(env, cfg1)
    Q2, r2 = train_mc_control(env, cfg2)
    # Compare final rolling mean of returns (rough signal)
    rm1 = float(np.mean(r1[-100:])) if len(r1) >= 100 else float(np.mean(r1))
    rm2 = float(np.mean(r2[-100:])) if len(r2) >= 100 else float(np.mean(r2))
    assert rm2 >= rm1 - 0.2  # allow noise; every-visit shouldn't be dramatically worse
