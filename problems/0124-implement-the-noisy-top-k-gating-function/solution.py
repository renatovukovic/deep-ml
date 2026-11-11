import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    noise_product = X @ W_noise
    softplus_output = np.log(1+np.exp(noise_product))
    #elementwise/hadamard product with N
    N_times_softplus = N * softplus_output
    gating_product = X @ W_g
    #add up
    H = gating_product + N_times_softplus
    #set everything except k largest entries per row to -inf
    # 1. Get the column indices of the top k elements per row
    top_k_idx = np.argpartition(H, -k, axis=1)[:, -k:]

    # 2. Create an output array filled with -np.inf
    result = np.full_like(H, -np.inf)

    # 3. Populate the top k values back into their original positions
    np.put_along_axis(result, top_k_idx, np.take_along_axis(H, top_k_idx, axis=1), axis=1)

    #finally softmax
    exps = np.exp(result - np.max(result, axis=1, keepdims=True))
    softmax = exps / np.sum(exps, axis=1, keepdims=True)

    return softmax