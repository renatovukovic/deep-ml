import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    states = list(range(transition_matrix.shape[0]))
    state_sequence = [initial_state]
    current_state = initial_state
    for i in range(num_steps):
        current_state = np.random.choice(states, p=transition_matrix[current_state])
        state_sequence.append(current_state)

    return np.array(state_sequence)
