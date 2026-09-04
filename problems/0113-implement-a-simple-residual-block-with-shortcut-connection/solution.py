import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:

	out = np.maximum(0, x @ w1.T)
	out = out @ w2.T
	return np.maximum(0, out + x) #maximum for elementwise max for relu
