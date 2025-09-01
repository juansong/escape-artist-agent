import numpy as np
import gymnasium as gym
from gymnasium import spaces

class EscapeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, n_traps=3):
        super().__init__()
        self.grid_size = grid_size
        self.n_traps = n_traps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        self.reward_goal = 1.0
        self.reward_trap = -1.0
        self.reward_step = -0.01
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0, 0)
        self.traps = []
        while len(self.traps) < self.n_traps:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos != self.state and pos not in self.traps:
                self.traps.append(pos)
        self.goal = (self.grid_size - 1, self.grid_size - 1)
        return self._state_to_int(self.state), {}

    def step(self, action):
        row, col = self.state
        if action == 0: row = max(row - 1, 0)
        elif action == 1: col = min(col + 1, self.grid_size - 1)
        elif action == 2: row = min(row + 1, self.grid_size - 1)
        elif action == 3: col = max(col - 1, 0)
        self.state = (row, col)

        if self.state == self.goal:
            reward = self.reward_goal
            done = True
        elif self.state in self.traps:
            reward = self.reward_trap
            done = True
        else:
            reward = self.reward_step
            done = False

        terminated = done
        truncated = False
        return self._state_to_int(self.state), reward, terminated, truncated, {}

    def _state_to_int(self, state):
        row, col = state
        return row * self.grid_size + col
