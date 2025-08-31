import numpy as np
import random
from collections import defaultdict

class MCAgent:
    """
    First-Visit Monte Carlo Control Agent for discrete action spaces.
    """

    def __init__(self, action_space, epsilon=0.1, gamma=0.99):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.epsilon = epsilon
        self.gamma = gamma

        # Q-value table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

        # Returns dictionary to store first-visit returns
        self.returns = defaultdict(lambda: [[] for _ in range(self.n_actions)])

    def choose_action(self, state):
        """ε-soft policy"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = self.Q[state]
            return int(np.argmax(q_values))

    def update(self, states, actions, rewards):
        """
        First-Visit Monte Carlo update.
        states, actions, rewards: lists from a complete episode
        """
        G = 0
        visited = set()

        # Traverse episode backward to compute return
        for t in reversed(range(len(states))):
            state, action, reward = states[t], actions[t], rewards[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                self.returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns[state][action])
                visited.add((state, action))
