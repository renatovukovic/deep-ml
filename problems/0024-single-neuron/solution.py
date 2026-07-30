import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here

	probabilities = np.dot(features, weights) + np.array(bias)

	#apply sigmoid
	probabilities = 1 / (1 + np.exp(-probabilities))

	#compute mse
	mse = np.mean((np.array(labels) - probabilities)**2) 


	return np.round(probabilities, 4).tolist(), round(float(mse), 4)