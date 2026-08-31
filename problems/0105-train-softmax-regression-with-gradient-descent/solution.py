import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	num_classes = max(y) + 1
	N = X.shape[0]

	# Add a column of ones as the first column
	X_bias = np.c_[np.ones(N), X]
	num_features = X_bias.shape[1]

	W = np.zeros((num_classes, num_features))

	losses = []

	#One-hot encode integer labels y into shape (N, C)
	Y_onehot = np.zeros((N, num_classes))
	Y_onehot[np.arange(N), y] = 1.0

	for i in range(iterations):

		out = X_bias @ W.T

		max_corrected_exp = np.exp(out - np.max(out, axis=1, keepdims=True))

		softmax_out = max_corrected_exp / np.sum(max_corrected_exp, axis=1, keepdims=True)

		# Adding 1e-15 inside log avoids numerical instability (log(0))
		loss = -np.sum(Y_onehot * np.log(softmax_out + 1e-15))
		losses.append(float(loss))

		#gradient of cross entropy is (y_true - y_pred) * x
		gradient = (softmax_out - Y_onehot).T @ X_bias

		W -= learning_rate * gradient




	return W.tolist(), losses
