import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	#compute query key product and before softmax add the mask
	q_times_k = np.dot(Q, K.T) / K.shape[1]
	#add the mask
	q_times_k += mask
	#softmax
	exp = np.exp(q_times_k - np.max(q_times_k, axis=-1, keepdims=True))
	softmax = exp / np.sum(exp, axis=-1, keepdims=True)

	#multiply with V
	return np.dot(softmax, V)
	