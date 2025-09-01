import os
import imageio
import numpy as np
from environment.escape_env import EscapeEnv
from agent.monte_carlo import MCAgent


def demo_episode(env, agent, max_steps=50, gif_path="demo.gif"):
    frames = []
    state = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        # Render environment as text (convert to image frame)
        frame = render_frame(env)
        frames.append(frame)

        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        step += 1

    # Add final frame
    frames.append(render_frame(env))

    # Save as GIF
    imageio.mimsave(gif_path, frames, duration=0.5)
    print(f"✅ Demo saved as {gif_path}")


def render_frame(env):
    """
    Render the current state of EscapeEnv into an image (numpy array).
    Uses matplotlib-like text rendering for simplicity.
    """
    import matplotlib.pyplot as plt

    grid = [["." for _ in range(env.grid_size)] for _ in range(env.grid_size)]

    # Place traps
    for trap in env.trap_positions:
        grid[trap[0]][trap[1]] = "X"

    # Place goal
    gx, gy = env.goal_pos
    grid[gx][gy] = "G"

    # Place agent
    ax, ay = env.state
    grid[ax][ay] = "A"

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    table = ax.table(
        cellText=grid,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)

    fig.canvas.draw()

    # Convert to numpy array
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return frame


if __name__ == "__main__":
    env = EscapeEnv(grid_size=5, trap_prob=0.2, seed=42)
    agent = MCAgent(env.action_space, epsilon=0.1, gamma=0.99)

    # Train briefly so the agent learns something
    for episode in range(200):
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

    # Run a demo and save it as a GIF
    demo_episode(env, agent, gif_path="escape_demo.gif")
