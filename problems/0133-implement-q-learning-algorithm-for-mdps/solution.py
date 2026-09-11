import numpy as np


def q_learning(
    num_states: int,
    num_actions: int,
    P: np.ndarray,  # P[s, a, s'] transition probability
    R: np.ndarray,  # R[s, a] or R[s, a, s'] reward matrix
    terminal_states: list,
    alpha: float,
    gamma: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:

    Q = np.zeros((num_states, num_actions))
    non_terminal_states = list(set(range(num_states)) - set(terminal_states))

    for e in range(num_episodes):
        current_state = np.random.choice(non_terminal_states)

        while True:
            # 1. Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # Uniform exploration
            else:
                action = np.argmax(Q[current_state])  # Greedy action

            # 2. Sample next state based on transition dynamics P[s, a, :]
            next_state = np.random.choice(
                num_states, p=P[current_state, action]
            )

            # Get reward (supports R[s, a] or R[s, a, s'])
            reward = (R[current_state, action])

            # 3. Target value for terminal vs non-terminal next_state
            if next_state in terminal_states:
                max_next_q = 0.0
            else:
                max_next_q = np.max(Q[next_state])

            # 4. Correct TD Update Formula
            td_target = reward + gamma * max_next_q
            td_error = td_target - Q[current_state, action]
            Q[current_state, action] += alpha * td_error

            # 5. Move to next state or end episode
            if next_state in terminal_states:
                break

            current_state = next_state

    return Q