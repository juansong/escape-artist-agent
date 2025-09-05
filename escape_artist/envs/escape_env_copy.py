# escape_artist/envs/escape_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    # Optional: use Gymnasium spaces if available (nicer repr/validation)
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback minimal shims
    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def contains(self, x) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

        def __repr__(self) -> str:
            return f"Discrete({self.n})"

    class _Box:
        def __init__(self, low, high, shape, dtype=np.int32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def contains(self, x) -> bool:
            arr = np.asarray(x)
            return arr.shape == tuple(self.shape)

        def __repr__(self) -> str:
            return f"Box(low={self.low}, high={self.high}, shape={self.shape}, dtype={self.dtype})"

    class _spaces:
        Discrete = _Discrete
        Box = _Box

    class _gym:
        spaces = _spaces

    spaces, gym = _spaces(), _gym()

from .generators import EMPTY, TRAP, GOAL, sample_trap_layout, neighbors_4


@dataclass
class EnvConfig:
    size: Tuple[int, int] = (8, 8)        # (height, width)
    start: Tuple[int, int] = (0, 0)       # (x, y)
    goal: Tuple[int, int] = (7, 7)        # (x, y)
    traps_pct: float = 0.08               # candidate density
    r_safe: int = 1                       # manhattan exclusion radius around start/goal
    slip: float = 0.0                     # probability action is replaced with random neighbor
    lethal_traps: bool = True
    step_cost: float = -0.01
    trap_penalty: float = -1.0
    goal_reward: float = 1.0
    max_steps: int = 200
    layout_mode: str = "per_episode"      # "per_env" | "per_episode"
    seed: Optional[int] = 42
    obs_mode: str = "pos"                 # "pos" | "pos_onehot" | "full_grid"


class EscapeEnv:
    """
    Escape Artist Agent environment (Gymnasium-style API).

    Observation modes:
      - "pos": 2D integer vector [x, y]
      - "pos_onehot": one-hot of the agent cell over H*W
      - "full_grid": HxW int grid (values {0=empty,1=trap,2=goal}) + agent pos returned in info
    """

    ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # up, right, down, left

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        self.h, self.w = self.cfg.size
        self.start = tuple(self.cfg.start)
        self.goal = tuple(self.cfg.goal)
        self._validate()

        # RNG
        self.base_seed = self.cfg.seed if self.cfg.seed is not None else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(self.base_seed)

        # episode state
        self.t = 0
        self.agent_pos = self.start
        self.episode_idx = 0

        # layout state
        self.grid = None  # np.ndarray (H, W) with {EMPTY, TRAP, GOAL}
        self._maybe_resample_layout(force=True)

        # Spaces
        self.action_space = spaces.Discrete(4)
        if self.cfg.obs_mode == "pos":
            self.observation_space = spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([self.w - 1, self.h - 1], dtype=np.int32),
                shape=(2,),
                dtype=np.int32,
            )
        elif self.cfg.obs_mode == "pos_onehot":
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.h * self.w,), dtype=np.int32
            )
        elif self.cfg.obs_mode == "full_grid":
            self.observation_space = spaces.Box(
                low=0, high=2, shape=(self.h, self.w), dtype=np.int8
            )
        else:
            raise ValueError(f"Unknown obs_mode={self.cfg.obs_mode}")

    # ---------- Gymnasium-like API ----------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Resets the environment.
        Returns: (obs, info)
        """
        if seed is not None:
            # reseed base_rng AND derive new deterministic episode stream
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)

        self.t = 0
        self.agent_pos = self.start

        if self.cfg.layout_mode == "per_episode":
            self._maybe_resample_layout(force=True)
        else:
            # per_env keeps the layout; still ensure we have one
            if self.grid is None:
                self._maybe_resample_layout(force=True)

        obs = self._obs()
        info = {
            "pos": self.agent_pos,
            "layout_id": self._layout_id(),
            "episode_idx": self.episode_idx,
        }
        self.episode_idx += 1
        return obs, info

    def step(self, action: int):
        """
        Steps the environment by one time step.
        Returns: (obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        ax, ay = self.agent_pos
        dx, dy = self._apply_action(action)

        # slip: with probability p, replace with random valid neighbor move
        if self.cfg.slip > 0.0 and self.rng.random() < self.cfg.slip:
            # pick a random neighbor that stays in bounds
            neighs = [(nx, ny) for nx, ny in neighbors_4(ax, ay, self.w, self.h)]
            nx, ny = neighs[self.rng.integers(0, len(neighs))]
        else:
            nx, ny = ax + dx, ay + dy
            nx = min(max(nx, 0), self.w - 1)
            ny = min(max(ny, 0), self.h - 1)

        self.agent_pos = (nx, ny)

        reward = self.cfg.step_cost
        terminated = False

        tile = self.grid[ny, nx]
        if tile == TRAP:
            reward = self.cfg.trap_penalty
            terminated = self.cfg.lethal_traps
        elif tile == GOAL:
            reward = self.cfg.goal_reward
            terminated = True

        self.t += 1
        truncated = self.t >= self.cfg.max_steps

        obs = self._obs()
        info = {
            "pos": self.agent_pos,
            "is_trap": tile == TRAP,
            "is_goal": tile == GOAL,
        }
        return obs, reward, terminated, truncated, info

    # ---------- Helpers ----------

    def _apply_action(self, a: int) -> Tuple[int, int]:
        """Map action to delta (dx, dy)."""
        return self.ACTIONS[int(a)]

    def _obs(self):
        if self.cfg.obs_mode == "pos":
            x, y = self.agent_pos
            return np.array([x, y], dtype=np.int32)
        elif self.cfg.obs_mode == "pos_onehot":
            x, y = self.agent_pos
            idx = y * self.w + x
            oh = np.zeros(self.h * self.w, dtype=np.int32)
            oh[idx] = 1
            return oh
        elif self.cfg.obs_mode == "full_grid":
            # NOTE: agent position is provided via info["pos"]
            return self.grid.copy()
        else:
            raise RuntimeError("invalid obs_mode (should be validated in __init__)")

    def _maybe_resample_layout(self, force: bool = False):
        if force or self.grid is None:
            # derive a deterministic episode-level seed for reproducibility
            ep_seed = int(np.uint32(hash((self.base_seed, self.episode_idx))))
            ep_rng = np.random.default_rng(ep_seed)

            grid = sample_trap_layout(
                rng=ep_rng,
                size=(self.h, self.w),
                start=self.start,
                goal=self.goal,
                traps_pct=self.cfg.traps_pct,
                r_safe=self.cfg.r_safe,
                max_resample=20,
            )
            self.grid = grid

    def _layout_id(self) -> str:
        # short hash of grid bytes for tagging
        return hex(np.uint32(np.frombuffer(self.grid.tobytes(), dtype=np.uint32).sum()))[2:]

    # ---------- Simple renderers ----------

    def render_ascii(self) -> str:
        """
        Returns an ASCII string representation (agent=@, trap=X, goal=G, empty=.)
        """
        chars = {0: ".", 1: "X", 2: "G"}
        lines = []
        ax, ay = self.agent_pos
        for y in range(self.h):
            row = []
            for x in range(self.w):
                if (x, y) == (ax, ay):
                    row.append("@")
                else:
                    row.append(chars.get(int(self.grid[y, x]), "?"))
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ---------- Convenience for compatibility ----------

    def seed(self, seed: Optional[int]):
        """Compatibility helper for older RL code."""
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        return [self.base_seed]

    # Optional: metadata for Gym render() API parity
    metadata = {"render_modes": ["ansi"]}
