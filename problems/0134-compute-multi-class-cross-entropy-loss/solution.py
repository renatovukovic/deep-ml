import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # 1. Clip probabilities to prevent log(0)
    preds_clipped = np.clip(predicted_probs, epsilon, 1 - epsilon)
    
    # 2. Calculate the loss matrix element-wise (only isolating the true classes)
    loss_matrix = -(true_labels * np.log(preds_clipped))
    
    # 3. Sum across classes (axis=-1), then average across the batch samples
    sample_losses = np.sum(loss_matrix, axis=-1)
    return float(np.mean(sample_losses))
