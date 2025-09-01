import pickle
from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent
import time
import os

def demo(grid_size=5):
    env = EscapeEnv(grid_size=grid_size)

    # Load Q-table
    with open("logs/q_table.pkl", "rb") as f:
        Q = pickle.load(f)

    agent = MCAgent(env.action_space)
    agent.Q = Q

    state, _ = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Simple text-based display
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]
        for trap in env.traps:
            grid[trap[0]][trap[1]] = "T"
        grid[env.goal[0]][env.goal[1]] = "G"
        row, col = divmod(state, grid_size)
        grid[row][col] = "A"

        os.system("clear")  # or "cls" on Windows
        for r in grid:
            print(" ".join(r))
        time.sleep(0.2)

    print("Demo finished!")

if __name__ == "__main__":
    demo()
