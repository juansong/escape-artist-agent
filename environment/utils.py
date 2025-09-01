import numpy as np
import json
import os

def load_map(file_path):
    """
    Load a grid map from a JSON or ASCII file.
    Returns a 2D list representing the grid.
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            grid = json.load(f)
    else:
        grid = []
        with open(file_path, "r") as f:
            for line in f:
                grid.append(list(line.strip()))
    return grid

def reward_shaping(state, goal, traps, step_penalty=-0.01, goal_reward=1.0, trap_penalty=-1.0):
    """
    Simple reward shaping function.
    """
    if state == goal:
        return goal_reward
    elif state in traps:
        return trap_penalty
    else:
        return step_penalty
