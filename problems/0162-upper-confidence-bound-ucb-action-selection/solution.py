import numpy as np

def ucb_action(counts, values, t, c):
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (np.ndarray): Number of times each action has been chosen
      values (np.ndarray): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    # TODO: Implement the UCB action selection
    #calculate the confidence bounds for each action
    best_action = 0
    best_ucb = 0
    for i, (count, reward) in enumerate(zip(counts, values)):
      
      ucb = reward + c * np.sqrt(np.log(t) / count)
      if ucb > best_ucb:
        best_ucb = ucb
        best_action = i

    return best_action