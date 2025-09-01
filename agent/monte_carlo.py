import numpy as np
from collections import defaultdict

class MCAgent:
    def __init__(self, action_space, epsilon=0.1, gamma=0.99):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.returns = defaultdict(list)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return int(np.argmax(self.Q[state]))

    def update(self, states, actions, rewards):
        G = 0
        visited = set()
        for t in reversed(range(len(states))):
            s, a = states[t], actions[t]
            r = rewards[t]
            G = self.gamma * G + r
            if (s, a) not in visited:
                self.returns[(s, a)].append(G)
                self.Q[s][a] = np.mean(self.returns[(s, a)])
                visited.add((s, a))
