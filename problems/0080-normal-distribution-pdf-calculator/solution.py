import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	
	normal_val = 1 / (math.sqrt(2*math.pi) * std_dev) * math.exp(-(1/2) * ((x - mean) / std_dev)**2)

	return round(normal_val,5)