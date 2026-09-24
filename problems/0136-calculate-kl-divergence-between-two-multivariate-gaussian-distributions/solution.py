import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    kl = 0.5 * (np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p)) - mu_p.shape[0] + (mu_q - mu_p).T @ np.linalg.inv(Cov_q) @ (mu_q - mu_p) + np.trace(np.linalg.inv(Cov_q) @ Cov_p))

    return kl
