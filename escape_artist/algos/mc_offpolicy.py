# escape_artist/algos/mc_offpolicy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
import json
import numpy as np

# (Optional) reuse the rollout helper for plots/eval
try:
    from .mc_control import greedy_policy_rollout  # noqa: F401
except Exception:
    greedy_policy_rollout = None  # not required for training


@dataclass
class OffMCConfig:
    episodes: int = 15_000
    gamma: float = 0.99
    is_type: str = "weighted"     # "weighted" | "ordinary"
    epsilon_behavior_start: float = 0.2
    epsilon_behavior_end: float = 0.02
    epsilon_behavior_decay_episodes: int = 10_000
    seed: Optional[int] = 42
    out_dir: Optional[str] = None  # saves Q, returns, params if set


# ---------- utilities ----------

def _epsilon_by_episode(start: float, end: float, horizon: int, ep: int) -> float:
    if horizon <= 0:
        return end
    frac = min(1.0, ep / horizon)
    return float(start + frac * (end - start))


def _state_to_index(pos: Tuple[int, int], w: int) -> int:
    x, y = pos
    return y * w + x


def _argmax_tie_break(row: np.ndarray, rng: np.random.Generator) -> int:
    m = row.max()
    cand = np.flatnonzero(row == m)
    return int(rng.choice(cand))


def _behavior_action_and_prob(
    Q: np.ndarray, s_idx: int, epsilon: float, rng: np.random.Generator
) -> Tuple[int, float]:
    """
    ε-greedy behavior policy with uniform tie-breaking among greedy actions.
    Returns (sampled_action, b_prob of that action).
    """
    nA = Q.shape[1]
    row = Q[s_idx]
    greedy_idxs = np.flatnonzero(row == row.max())
    k = len(greedy_idxs)

    if rng.random() < epsilon:
        a = int(rng.integers(0, nA))
    else:
        a = int(rng.choice(greedy_idxs))

    # probability of chosen action under ε-greedy with uniform tie handling
    if a in greedy_idxs:
        b_prob = epsilon / nA + (1.0 - epsilon) * (1.0 / k)
    else:
        b_prob = epsilon / nA
    return a, float(b_prob)


def _ensure_out_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save_artifacts(out_dir: str, Q: np.ndarray, returns: List[float], cfg: OffMCConfig):
    np.save(os.path.join(out_dir, "Q.npy"), Q)
    np.save(os.path.join(out_dir, "returns.npy"), np.asarray(returns, dtype=np.float32))
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(
            {
                "episodes": cfg.episodes,
                "gamma": cfg.gamma,
                "is_type": cfg.is_type,
                "epsilon_behavior_start": cfg.epsilon_behavior_start,
                "epsilon_behavior_end": cfg.epsilon_behavior_end,
                "epsilon_behavior_decay_episodes": cfg.epsilon_behavior_decay_episodes,
                "seed": cfg.seed,
            },
            f,
            indent=2,
        )


# ---------- episode generation (behavior policy) ----------

def generate_episode_offpolicy(
    env,
    Q: np.ndarray,
    epsilon_behavior: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[float], List[float]]:
    """
    Roll out one episode using ε-greedy *behavior* policy over Q.
    Returns (state_indices, actions, rewards, behavior_action_probs).
    """
    obs, info = env.reset()
    h, w = env.h, env.w
    s_list: List[int] = []
    a_list: List[int] = []
    r_list: List[float] = []
    bprob_list: List[float] = []

    while True:
        # robust state indexing (env puts (x,y) in info["pos"])
        if "pos" in info:
            s_idx = _state_to_index(info["pos"], w)
        else:
            x, y = int(obs[0]), int(obs[1])
            s_idx = _state_to_index((x, y), w)

        a, bprob = _behavior_action_and_prob(Q, s_idx, epsilon_behavior, rng)
        obs, r, term, trunc, info = env.step(a)

        s_list.append(s_idx)
        a_list.append(a)
        r_list.append(float(r))
        bprob_list.append(bprob)

        if term or trunc:
            return s_list, a_list, r_list, bprob_list


# ---------- training: off-policy MC control ----------

def train_mc_offpolicy(env, cfg: OffMCConfig) -> Tuple[np.ndarray, List[float]]:
    """
    Off-policy Monte Carlo control with importance sampling.
    Target π is *greedy* w.r.t Q (deterministic).
    Behavior b is ε-greedy w.r.t Q with decaying ε.

    is_type = "weighted" uses incremental weighted-IS update:
        C[s,a] += W
        Q[s,a] += (W / C[s,a]) * (G - Q[s,a])

    is_type = "ordinary" uses ordinary-IS incremental mean over (W*G):
        N[s,a] += 1
        Q[s,a] += (W*G - Q[s,a]) / N[s,a]

    As in Sutton & Barto, the update proceeds backward through the episode
    and *breaks* once the taken action is not greedy under the current Q.

    Args:
        env (EscapeEnv).
        cfg (OffMCConfig).
    Returns:
        Q (np.ndarray): shape (H*W,4).
        returns (np.ndarray[episodes]).
    """
    assert cfg.is_type in ("weighted", "ordinary")

    rng = np.random.default_rng(cfg.seed)

    h, w = env.h, env.w
    nS, nA = h * w, getattr(env.action_space, "n", 4)

    Q = np.zeros((nS, nA), dtype=np.float32)
    episodic_returns: List[float] = []

    # accumulators
    if cfg.is_type == "weighted":
        C = np.zeros((nS, nA), dtype=np.float64)  # cumulative weights
    else:
        N = np.zeros((nS, nA), dtype=np.int32)    # sample counts (ordinary IS)

    for ep in range(cfg.episodes):
        eps_b = _epsilon_by_episode(
            cfg.epsilon_behavior_start, cfg.epsilon_behavior_end,
            cfg.epsilon_behavior_decay_episodes, ep
        )

        s_list, a_list, r_list, bprob_list = generate_episode_offpolicy(env, Q, eps_b, rng)
        # behavior episodic return (for logging curve)
        episodic_returns.append(float(sum(r_list)))

        # compute return suffixes backwards
        G = 0.0
        W = 1.0
        for t in reversed(range(len(r_list))):
            s_t = s_list[t]
            a_t = a_list[t]
            r_tp1 = r_list[t]
            b_prob = bprob_list[t]

            G = cfg.gamma * G + r_tp1

            if cfg.is_type == "weighted":
                # weighted importance sampling
                C[s_t, a_t] += W
                # incremental update towards G with step (W / C)
                Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            else:
                # ordinary importance sampling
                N[s_t, a_t] += 1
                Q[s_t, a_t] += (W * G - Q[s_t, a_t]) / N[s_t, a_t]

            # improve target policy to be greedy at s_t
            greedy_a = _argmax_tie_break(Q[s_t], rng)
            # if the taken action isn't greedy under the target, π(a|s)=0 ⇒ break
            if a_t != greedy_a:
                break

            # continue weighting for earlier state-action pairs
            if b_prob <= 0.0:
                break  # safety
            W = W / b_prob

        # light progress print
        if (ep + 1) % max(1, cfg.episodes // 20) == 0:
            print(
                f"[MC-OFF/{cfg.is_type}] ep {ep + 1}/{cfg.episodes} "
                f"| eps_b={eps_b:.3f} | G_b={episodic_returns[-1]:.3f}"
            )

    # save
    out_dir = _ensure_out_dir(cfg.out_dir)
    if out_dir:
        _save_artifacts(out_dir, Q, episodic_returns, cfg)
        if cfg.seed is not None:
            with open(os.path.join(out_dir, "seed.txt"), "w") as f:
                f.write(str(cfg.seed))

    return Q, episodic_returns
