import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    # Your code here
    
    #update the moments
    m_new = beta1 * m + (1-beta1) * g
    v_new = beta2 * v + (1-beta2) * g**2

    #apply bias correction
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)

    #parameter update with weight decay
    w_new = w - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * w)

    return w_new, m_new, v_new

