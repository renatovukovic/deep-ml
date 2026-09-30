import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    new_V = np.zeros_like(V)
    
    # Loop through each state s
    for s in range(len(V)):
        action_values = []
        
        # Look at all actions available in state s
        for a, outcomes in transitions[s].items():
            action_value = 0
            
            # Sum over all possible next states s' for action a
            for prob, next_state, reward, done in outcomes:
                # If the episode is done, future value is 0
                next_val = 0 if done else V[next_state]
                action_value += prob * (reward + gamma * next_val)
                
            action_values.append(action_value)
            
        # The new value is the maximum expected return across all actions
        # If a state has no valid actions, its value remains 0
        new_V[s] = max(action_values) if action_values else 0.0
        
    return new_V
