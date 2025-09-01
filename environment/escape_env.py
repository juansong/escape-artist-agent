import gym
from gym import spaces
import numpy as np
import random

class EscapeEnv(gym.Env):
    """
    Escape Grid-World Environment.
    The agent must reach the exit (E) while avoiding traps (T).
    
    Tiles:
    - S : Start (safe)
    - F : Floor (safe)
    - T : Trap (danger, episode ends)
    - E : Exit / Goal (success, episode ends)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=5):
        super(EscapeEnv, self).__init__()

        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation space: agent position (row, col) in grid
        self.observation_space = spaces.Tuple((
            spaces.Discrete(grid_size),
            spaces.Discrete(grid_size)
        ))

        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset environment at the start of an episode"""
        # Initialize map with safe floor
        self.map = np.full((self.grid_size, self.grid_size), "F")

        # Place Start (S) and Exit (E)
        self.start = (0, 0)
        self.exit = (self.grid_size - 1, self.grid_size - 1)
        self.map[self.start] = "S"
        self.map[self.exit] = "E"

        # Randomly place traps (T), avoiding start and exit
        for _ in range(int(self.grid_size * 1.5)):
            r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (r, c) not in [self.start, self.exit]:
                self.map[r, c] = "T"

        # Initialize agent position
        self.agent_pos = list(self.start)

        return tuple(self.agent_pos)

    def step(self, action):
        """Perform an action and return new state, reward, done, info"""
        row, col = self.agent_pos

        # Move agent
        if action == 0:   # up
            row = max(row - 1, 0)
        elif action == 1: # down
            row = min(row + 1, self.grid_size - 1)
        elif action == 2: # left
            col = max(col - 1, 0)
        elif action == 3: # right
            col = min(col + 1, self.grid_size - 1)

        self.agent_pos = [row, col]

        tile = self.map[row, col]
        reward, done = -0.01, False  # small step penalty

        if tile == "T":  # fell into a trap
            reward, done = -1.0, True
        elif tile == "E":  # reached exit
            reward, done = 1.0, True

        return tuple(self.agent_pos), reward, done, {}

    def render(self, mode="human"):
        """Print the grid with the agent's position"""
        grid = self.map.copy()
        r, c = self.agent_pos
        grid[r, c] = "A"  # mark agent's position
        print("\n".join([" ".join(row) for row in grid]))
        print()
