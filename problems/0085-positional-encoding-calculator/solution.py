import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    Computes Sinusoidal Positional Encodings for a sequence.

    :param position: Total sequence length N
    :param d_model: Dimensionality of the embedding space
    :return: Array of shape (position, d_model)
    """
    # 1. Create positions array of shape (position, 1)
    pos_indices = np.arange(position)[:, np.newaxis]  # Shape: (position, 1)
    
    # 2. Compute frequency division terms for even indices 2i
    i = np.arange(0, d_model, 2)  # Even dimension indices
    div_term = 1.0 / (10000.0 ** (i / d_model))
    
    # 3. Calculate phase angles (No division by 2)
    angles = pos_indices * div_term  # Shape: (position, ceil(d_model / 2))
    
    # 4. Initialize result matrix
    res = np.zeros((position, d_model), dtype=np.float32)
    
    # 5. Interleave Sines into even indices and Cosines into odd indices
    res[:, 0::2] = np.sin(angles)
    res[:, 1::2] = np.cos(angles[:, :d_model // 2])  # Safely handles odd d_model
    
    return res.astype(np.float16)