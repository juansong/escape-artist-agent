import numpy as np

def epsilon_greedy(Q_s, epsilon=0.1):
    """
    Epsilon-greedy policy for selecting actions
    Q_s: array of action values for the current state
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q_s))
    else:
        return int(np.argmax(Q_s))

def greedy(Q_s):
    """Select the action with highest Q-value."""
    return int(np.argmax(Q_s))
