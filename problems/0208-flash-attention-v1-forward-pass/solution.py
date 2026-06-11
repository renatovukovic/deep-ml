import numpy as np


def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """
    Flash Attention v1 (CPU / NumPy version).

    Q, K, V:
        (N, d)

    Returns:
        O: (N, d)
    """

    N, d = Q.shape

    O = np.zeros((N, d), dtype=np.float32)

    # running max
    m = np.full(N, -np.inf, dtype=np.float32)

    # running softmax denominator
    l = np.zeros(N, dtype=np.float32)

    scale = 1.0 / np.sqrt(d)

    # iterate over K/V blocks
    for j in range(0, N, block_size):

        Kj = K[j:j + block_size]
        Vj = V[j:j + block_size]

        # scores for ALL queries against current key block
        S = (Q @ Kj.T) * scale
        # shape: (N, B)

        # local max in this block
        m_block = np.max(S, axis=1)

        # updated global max
        m_new = np.maximum(m, m_block)

        # rescale previous statistics
        alpha = np.exp(m - m_new)

        # exponentials in current block
        P = np.exp(S - m_new[:, None])

        # updated denominator
        l_new = alpha * l + np.sum(P, axis=1)

        # update output numerator
        O = (
            (alpha * l)[:, None] * O
            + P @ Vj
        ) / l_new[:, None]

        m = m_new
        l = l_new

    return O