import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)

    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    softmax_input = np.dot(Q, K.T) / np.sqrt(K.shape[1])
    exp = np.exp(softmax_input - np.max(softmax_input, axis=-1, keepdims=True))
    softmax = exp / np.sum(exp, axis=-1, keepdims=True)

    return np.dot(softmax, V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    
    Q_heads = np.array_split(Q, n_heads, axis=-1)
    K_heads = np.array_split(K, n_heads, axis=-1)
    V_heads = np.array_split(V, n_heads, axis=-1)

    #apply self attention
    attention_out = []
    for i in range(n_heads):
        attention_out.append(self_attention(Q_heads[i], K_heads[i], V_heads[i]))

    
    #return the concatenated
    return np.concatenate(attention_out, axis=-1)

