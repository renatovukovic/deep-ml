import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    # Your code here
    if np.random.rand() < epsilon:
        return np.random.choice(len(Q))
   
    # 2. Exploitation: pick among actions that tie for max Q-value
    max_val = np.max(Q)
    max_actions = np.flatnonzero(Q == max_val)

    return int(np.random.choice(max_actions))