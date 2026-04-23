import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = K.shape[1]
    qk = np.dot(Q, K.T)
    qk_normalised = qk / np.sqrt(d_k)
    #compute softmax of this term
    exps = np.exp(qk_normalised - np.max(qk_normalised, axis=-1, keepdims=True)) #subtract max for numerical stability
    softmax = exps / np.sum(exps, axis=-1, keepdims=True)
    #multiply with V
    return np.dot(softmax, V)