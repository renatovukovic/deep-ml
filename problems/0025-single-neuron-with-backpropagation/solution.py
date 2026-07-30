import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here

	mse_values = []

	for i in range(epochs):

		probabilities = np.dot(features, initial_weights) + initial_bias

		#apply sigmoid
		probabilities = 1 / (1 + np.exp(-probabilities))

		#compute mse
		mse = np.mean((labels - probabilities)**2) 

		mse_values.append(float(mse))

		errors = probabilities - labels
		# 3. Compute gradients via chain rule
		# dL/dz = 2/N * error * sigmoid_derivative
		sigmoid_deriv = probabilities * (1 - probabilities)
		dz = (2 / len(labels)) * errors * sigmoid_deriv

		dw = np.dot(features.T, dz)
		db = np.sum(dz)

		initial_weights -= learning_rate * dw
		initial_bias -= learning_rate *db



	return initial_weights, initial_bias, mse_values