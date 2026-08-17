
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	#r_squared is 1 - regression sum of squares / total sum of squares

	regression_sum_squares = sum((y_true-y_pred)**2)
	total_sum_squares = sum((y_true - np.mean(y_true))**2)

	return 1 - regression_sum_squares / total_sum_squares
	
