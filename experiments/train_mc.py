import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent

def train_mc(episodes=5000, gamma=0.99, epsilon=0.1, log_dir="logs"):
    env = EscapeEnv(grid_size=5)
    agent = MCAgent(env.action_space, epsilon=epsilon, gamma=gamma)

    rewards_per_episode = []

    for episode in tqdm(range(episodes), desc="Training"):
        states, actions, rewards = [], [], []
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        agent.update(states, actions, rewards)
        rewards_per_episode.append(np.sum(rewards))

    os.makedirs(log_dir, exist_ok=True)

    # Save Q-table (convert defaultdict to dict)
    q_table_path = os.path.join(log_dir, "q_table.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(dict(agent.Q), f)

    # Save training rewards
    log_path = os.path.join(log_dir, "training_log.csv")
    df = pd.DataFrame({
        "episode": np.arange(1, episodes + 1),
        "reward": rewards_per_episode
    })
    df.to_csv(log_path, index=False)

    print(f"Training finished. Logs saved in '{log_dir}'")
    return rewards_per_episode

if __name__ == "__main__":
    train_mc()
