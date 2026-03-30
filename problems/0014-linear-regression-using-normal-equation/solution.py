import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	#normal equation: theta = (xT * x)**-1 * xT * y

	#turn list to np arrays
	x = np.array(X)
	y = np.array(y)

	xTx = np.dot(x.T, x)
	xTx_inverse = np.linalg.inv(xTx)
	xTy = np.dot(x.T, y)

	theta = np.dot(xTx_inverse, xTy)


	theta = [round(feat, 4) for feat in theta]

	return theta