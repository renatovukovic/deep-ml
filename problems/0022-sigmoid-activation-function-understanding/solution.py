import math

def sigmoid(z: float) -> float:
	#Your code here

	#sigmoid function is 1 / 1 + exp(-z)

	result = round(1 / (1 + math.exp(-z)), 4)

	return result