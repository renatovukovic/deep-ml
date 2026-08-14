import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):


	#use formula for directly computing KL divergence from two normal means and variances
	#kl = log(sigma_p/sigma_q) + (sigma_q + (mu_p-mu_q)^2) / s sigma_q^2 - 1/2
	kl_divergence = np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2*sigma_q**2) - 0.5

	return kl_divergence
