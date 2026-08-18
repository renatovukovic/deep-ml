
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here

	subtraction = np.array(y_true) - np.array(y_pred)
	squared = np.power(subtraction, 2)
	mean = np.mean(squared)
	rmse_res = np.sqrt(mean)


	return round(rmse_res,3)
