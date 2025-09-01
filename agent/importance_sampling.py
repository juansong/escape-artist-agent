import numpy as np

class OffPolicyMCAgent:
    """
    Off-policy Monte Carlo agent using Weighted Importance Sampling.
    """
    def __init__(self, action_space, gamma=0.99):
        self.action_space = action_space
        self.gamma = gamma
        self.Q = {}  # {(state, action): value}
        self.C = {}  # cumulative weights for importance sampling

    def update(self, episode):
        G = 0
        W = 1.0
        for state, action, reward, pi_prob, b_prob in reversed(episode):
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] = self.Q.get(key, 0.0)
            self.C[key] = self.C.get(key, 0.0)
            self.C[key] += W
            self.Q[key] += (W / self.C[key]) * (G - self.Q[key])
            W *= pi_prob / b_prob
            if W == 0:
                break
