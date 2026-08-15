import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here


	final_hidden_state = np.array(initial_hidden_state, dtype=np.float32)
	Wx = np.array(Wx, dtype=np.float32)
	Wh = np.array(Wh, dtype=np.float32)
	b = np.array(b, dtype=np.float32)

	for x in input_sequence:
		#compute new hidden state
		current_hidden = Wh @ final_hidden_state
		current_input_projection = Wx @ np.array(x, dtype=np.float32) + b

		final_hidden_state = np.tanh(current_hidden + current_input_projection)

	return final_hidden_state