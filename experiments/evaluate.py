import pickle
import numpy as np
from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent

def evaluate(env, agent, episodes=100):
    success = 0
    steps_list = []
    detection = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if reward < 0:
                detection += 1

        if reward > 0:
            success += 1
        steps_list.append(steps)

    print(f"Evaluation over {episodes} episodes:")
    print(f"Success Rate: {success/episodes*100:.2f}%")
    print(f"Average Steps: {np.mean(steps_list):.2f}")
    print(f"Detection Rate: {detection/episodes*100:.2f}%")

if __name__ == "__main__":
    env = EscapeEnv(grid_size=5)

    # Load Q-table
    with open("logs/q_table.pkl", "rb") as f:
        Q = pickle.load(f)

    agent = MCAgent(env.action_space)
    agent.Q = Q

    evaluate(env, agent)
