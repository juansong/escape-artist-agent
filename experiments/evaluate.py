import argparse
import pickle
import numpy as np
from tqdm import tqdm

from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent


def evaluate_mc(model_path="saved_models/mc_policy.pkl", episodes=500):
    env = EscapeEnv(grid_size=5)

    # Load trained policy (Q-table)
    with open(model_path, "rb") as f:
        Q = pickle.load(f)

    agent = MCAgent(env.action_space)
    agent.Q = Q  # assign learned Q-table

    successes, steps, traps = 0, [], 0

    for _ in tqdm(range(episodes), desc="Evaluating"):
        state = env.reset()
        done, step_count = False, 0

        while not done:
            # Greedy action selection (no epsilon exploration)
            action = np.argmax([agent.Q.get((state, a), 0.0) for a in range(env.action_space.n)])
            state, reward, done, _ = env.step(action)
            step_count += 1

            if reward == 1.0:
                successes += 1
            elif reward == -1.0:
                traps += 1

        steps.append(step_count)

    success_rate = successes / episodes
    avg_steps = np.mean(steps)
    trap_rate = traps / episodes

    print("\n--- Evaluation Results ---")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Trap Encounter Rate: {trap_rate * 100:.2f}%")

    return success_rate, avg_steps, trap_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Monte Carlo policy")
    parser.add_argument("--model", type=str, default="saved_models/mc_policy.pkl", help="Path to trained policy")
    parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes")

    args = parser.parse_args()
    evaluate_mc(args.model, args.episodes)
