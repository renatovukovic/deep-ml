import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
	moment1_ema = np.zeros_like(x0)
	moment2_ema = np.zeros_like(x0)

	for epoch in range(1,num_iterations+1):
		gradient = grad(x0)
		moment1_ema = beta1 * moment1_ema + (1-beta1) * gradient
		moment2_ema = beta2 * moment2_ema + (1-beta2) * gradient**2

		#bias correction since emas were initialised at zero
		moment1 = moment1_ema / (1 - beta1**epoch)
		moment2 = moment2_ema / (1 - beta2**epoch)


		#update x0
		step = learning_rate / (np.sqrt(moment2) + epsilon) * moment1 
		x0 = x0 - step

	
	return x0

