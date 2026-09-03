import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    
    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics
    
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    
    if not training: #only apply statistics and do not compute them
        X_normalised = (X - running_mean) / (np.sqrt(running_var + epsilon))
        X_normalised = gamma * X_normalised + beta
        return X_normalised, running_mean, running_var
    
    else:

        if running_mean is None:
            #initialise to zero 
            running_mean = np.zeros((1,X.shape[1],1,1))
        if running_var is None:
            #initialise to zero 
            running_var = np.ones((1,X.shape[1],1,1))
        #compute mean and variance
        current_mean = np.mean(X, axis=(0,2,3), keepdims=True)
        current_var = np.var(X, axis=(0,2,3), keepdims=True)

        X_normalised = (X - current_mean) / (np.sqrt(current_var + epsilon))
        X_normalised = gamma * X_normalised + beta

        #update running metrics
        running_mean = (1-momentum) * running_mean + momentum * current_mean
        running_var = (1-momentum) * running_var + momentum * current_var

        return X_normalised, running_mean, running_var


