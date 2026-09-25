import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Split the input into two equal parts of size d along the last axis
    x1, x2 = np.split(x, 2, axis=-1)
    
    # Calculate Swish (SiLU) activation for the gate path: x * sigmoid(x)
    sigmoid_x1 = 1 / (1 + np.exp(-x1))
    swish_x1 = x1 * sigmoid_x1

    sigmoid_x2 = 1 / (1 + np.exp(-x2))
    swish_x2 = x2 * sigmoid_x2
    
    # Element-wise multiplication between the gated path and the value path
    scores = x1 * swish_x2
    
    return scores
