# escape_artist/envs/escape_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Cell types
EMPTY = 0
TRAP  = 1
GOAL  = 2

# Simple Discrete action space (to avoid gym dependency)
class Discrete:
    def __init__(self, n: int):
        self.n = int(n)

@dataclass
class EnvConfig:
    size: Tuple[int, int] = (8, 8)      # (H, W)
    start: Tuple[int, int] = (0, 0)     # (x, y)
    goal: Tuple[int, int]  = (7, 7)     # (x, y)
    traps_pct: float = 0.08
    r_safe: int = 1
    slip: float = 0.0                   # prob to replace action with random direction
    lethal_traps: bool = True
    step_cost: float = -0.01
    trap_penalty: float = -1.0
    goal_reward: float = 1.0
    max_steps: int = 200
    layout_mode: str = "per_episode"    # "per_env" | "per_episode"
    seed: Optional[int] = 42
    obs_mode: str = "pos"               # currently supports "pos"

# Generators utilities imported for sampling
from .generators import sample_trap_layout, TRAP, GOAL

class EscapeEnv:
    """
    Minimal Gymnasium-style environment for the Escape Artist gridworld.

    Args(constructor):
        cfg: EnvConfig

    Attributes: 
        h, w: int, grid: np.ndarray, agent_pos: (x,y), action_space.n==4, _layout_id: int

    Methods:
        _validate() — internal assertions (sizes, bounds, ranges)
        _generate_layout() — internal; creates grid via sample_trap_layout.
        _obs_from_pos(pos) — returns observation [x, y] as np.int32.
    """
    metadata: Dict[str, Any] = {}

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.h, self.w = int(cfg.size[0]), int(cfg.size[1])
        self.action_space = Discrete(4)
        self.rng = np.random.default_rng(cfg.seed)
        self._episode_idx = 0

        # internal state
        self.grid = None  # np.ndarray (H,W)
        self.agent_pos = None  # (x,y)
        self._layout_id = None

        self._validate()
        # Initialize layout for per_env so layout_id is stable across resets
        if self.cfg.layout_mode == "per_env":
            self._generate_layout()

    # ------------------------------- helpers -------------------------------
    def _validate(self):
        assert self.h > 0 and self.w > 0, "Grid size must be positive"
        sx, sy = self.cfg.start
        gx, gy = self.cfg.goal
        assert 0 <= sx < self.w and 0 <= sy < self.h, "Start out of bounds"
        assert 0 <= gx < self.w and 0 <= gy < self.h, "Goal out of bounds"
        assert (sx, sy) != (gx, gy), "Start and goal must differ"
        assert 0.0 <= self.cfg.traps_pct < 1.0, "traps_pct in [0,1)"
        assert self.cfg.layout_mode in ("per_env", "per_episode"), "layout_mode invalid"
        assert 0.0 <= self.cfg.slip <= 1.0, "slip must be in [0,1]"

    def _generate_layout(self):
        # Seed per layout for reproducibility using base seed + episode index
        base_seed = None if self.cfg.seed is None else int(self.cfg.seed)
        local_seed = None if base_seed is None else base_seed + self._episode_idx * 997
        rng = np.random.default_rng(local_seed)
        self.grid = sample_trap_layout(
            rng=rng,
            size=(self.h, self.w),
            start=self.cfg.start,
            goal=self.cfg.goal,
            traps_pct=self.cfg.traps_pct,
            r_safe=self.cfg.r_safe,
            max_resample=50,
        )
        # deterministic-ish ID for this layout for testing
        self._layout_id = int(np.uint32(self.grid.sum() * 2654435761) % (2**31 - 1))

    def _obs_from_pos(self, pos: Tuple[int,int]) -> np.ndarray:
        if self.cfg.obs_mode == "pos":
            return np.array([pos[0], pos[1]], dtype=np.int32)
        # Fallback to pos for any other mode to keep algos working
        return np.array([pos[0], pos[1]], dtype=np.int32)

    # ------------------------------- API -------------------------------
    def reset(self, seed: Optional[int] = None):
        '''
        Creates a fresh layout if layout_mode="per_episode" (or first call). Resets agent to start.
        info contains "pos", "is_trap", "is_goal", "layout_id".

        Args:
            seed (int|None): optional RNG reseed for this reset.
            
        Returns:
            info (dict): {"pos": (x,y), "is_trap": bool, "is_goal": bool, "layout_id": int}.
        '''
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # New layout each episode in per_episode mode
        if self.cfg.layout_mode == "per_episode" or self.grid is None:
            self._generate_layout()

        self.agent_pos = tuple(self.cfg.start)
        self._steps = 0
        self._episode_idx += 1
        obs = self._obs_from_pos(self.agent_pos)
        info = {
            "pos": self.agent_pos,
            "is_trap": False,
            "is_goal": False,
            "layout_id": self._layout_id,
        }
        return obs, info

    def step(self, action: int):
        '''
        Clips movement to grid. Applies slip; sets trap/goal flags.

        Args:
            action (int): {0:up, 1:right, 2:down, 3:left} (with slip applied).
        Returns:
            obs (np.ndarray[int32]): [x, y] after move.
            reward (float): step_cost + (optional penalties/rewards).
            terminated (bool): reached goal or lethal trap.
            truncated (bool): time limit exceeded.
            info (dict): includes "pos", "is_trap", "is_goal", "layout_id".
        '''
        self._steps += 1
        # slip handling
        if self.rng.random() < self.cfg.slip:
            action = int(self.rng.integers(0, 4))

        # map action
        dx = {0: 0, 1: 1, 2: 0, 3: -1}
        dy = {0: -1, 1: 0, 2: 1, 3: 0}
        x, y = self.agent_pos
        xn = int(np.clip(x + dx.get(action, 0), 0, self.w - 1))
        yn = int(np.clip(y + dy.get(action, 0), 0, self.h - 1))
        self.agent_pos = (xn, yn)

        cell = self.grid[yn, xn]
        is_trap = cell == TRAP
        is_goal = cell == GOAL

        reward = self.cfg.step_cost
        terminated = False
        truncated = False

        if is_trap:
            reward += self.cfg.trap_penalty
            if self.cfg.lethal_traps:
                terminated = True
        if is_goal:
            reward += self.cfg.goal_reward
            terminated = True

        if self._steps >= self.cfg.max_steps and not terminated:
            truncated = True

        obs = self._obs_from_pos(self.agent_pos)
        info = {
            "pos": self.agent_pos,
            "is_trap": bool(is_trap),
            "is_goal": bool(is_goal),
            "layout_id": self._layout_id,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info
