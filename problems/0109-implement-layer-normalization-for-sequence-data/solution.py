import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	normalise by subtracting mean for centering then dividing by the sqrt of( variance and adding epsilon for numerical stability) i.e. std plus epsilon
	for the final output we multiply with gamma and add beta so that the model can learn how much it wants to use the normalisation
	"""
	# 1. Calculate Mean across the last dimension (hidden_dim)
	# keepdims=True ensures the shape is (batch_size, seq_len, 1) for broadcasting
	X_mean = np.mean(X, axis=-1, keepdims=True)

	# 2. Calculate Variance across the last dimension
	X_var = np.var(X, axis=-1, keepdims=True)

	# 3. Standardize: Center and divide by standard deviation
	centred = X - X_mean
	normalised = centred / np.sqrt(X_var + epsilon)

	# 4. Scale and Shift (Affinity)
	# gamma and beta will broadcast across the batch and sequence dimensions
	out = gamma * normalised + beta

	return out