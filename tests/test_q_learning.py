# tests/test_q_learning.py
from escape_artist.envs.escape_env import EscapeEnv, EnvConfig
from escape_artist.algos.q_learning import QLConfig, train_q_learning
from escape_artist.algos.mc_control import greedy_policy_rollout  # reuse helper

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
        seed=321,
        obs_mode="pos",
    )
    return EscapeEnv(cfg)

def test_q_learning_reaches_goal_easy():
    env = make_easy_env()
    q_cfg = QLConfig(
        episodes=3000, gamma=0.99,
        alpha_start=0.5, alpha_end=0.1, alpha_decay_episodes=2500,
        epsilon_start=0.2, epsilon_end=0.02, epsilon_decay_episodes=2500,
        seed=7,
    )
    Q, returns = train_q_learning(env, q_cfg)
    total, traj = greedy_policy_rollout(env, Q, max_steps=env.cfg.max_steps)
    assert total > 0.5
    assert 1 < len(traj) <= env.cfg.max_steps
