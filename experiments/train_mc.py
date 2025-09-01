import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent


def train_mc(episodes=5000, gamma=0.99, epsilon=0.1, save_path="saved_models/mc_policy.pkl"):
    env = EscapeEnv(grid_size=5)
    agent = MCAgent(env.action_space, epsilon=epsilon, gamma=gamma)

    rewards_per_episode = []

    for episode in tqdm(range(episodes), desc="Training"):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        agent.update(states, actions, rewards)
        rewards_per_episode.append(np.sum(rewards))

    # Save trained policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(agent.Q, f)

    print(f"\nTraining finished. Policy saved to {save_path}")
    return rewards_per_episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Monte Carlo Agent in EscapeEnv")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--save_path", type=str, default="saved_models/mc_policy.pkl", help="Path to save policy")

    args = parser.parse_args()
    train_mc(args.episodes, args.gamma, args.epsilon, args.save_path)
