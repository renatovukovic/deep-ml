import numpy as np


def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """Perform one step of gradient boosting regression using a decision stump.

    Args:
        X: Feature matrix (list of lists or np.ndarray), shape (n_samples,
          n_features)
        y: Target values (list or np.ndarray), shape (n_samples,)
        current_predictions: Current ensemble predictions (list or np.ndarray),
          shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)

    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    # Convert inputs to NumPy arrays for efficient vector and matrix manipulation
    X = np.array(X)
    y = np.array(y)
    current_predictions = np.array(current_predictions)

    n_samples, n_features = X.shape

    # 1. Compute the residuals (negative gradient for MSE loss: true_y - pred)
    residuals = y - current_predictions

    # 2. Set up baseline/fallback values if no beneficial split is found
    best_mse = float("inf")
    default_pred = np.mean(residuals)
    best_stump_pred = np.full(n_samples, default_pred)

    # Track parameters for the optimal split
    best_feature = None
    best_threshold = None
    best_left_val = None
    best_right_val = None

    # 3. Exhaustively search across all features and candidate thresholds
    for col in range(n_features):
        feature_vals = X[:, col]
        # Extract unique values and sort them to find midpoints
        unique_vals = np.unique(feature_vals)

        if len(unique_vals) <= 1:
            continue  # Cannot split if all values for this feature are identical

        # Candidate thresholds are midpoints between consecutive unique values
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            # Partition the samples into left (<= threshold) and right (> threshold)
            left_mask = feature_vals <= threshold
            right_mask = feature_vals > threshold

            # Skip split if it leaves either partition empty
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            # Leaf values are the mean of residuals within each partition
            left_residuals = residuals[left_mask]
            right_residuals = residuals[right_mask]

            left_val = np.mean(left_residuals)
            right_val = np.mean(right_residuals)

            # Compute weighted Mean Squared Error (MSE) for this split
            left_mse = np.mean((left_residuals - left_val) ** 2)
            right_mse = np.mean((right_residuals - right_val) ** 2)
            total_mse = (
                len(left_residuals) * left_mse
                + len(right_residuals) * right_mse
            ) / n_samples

            # Keep track of the split if it minimizes the MSE further
            if total_mse < best_mse:
                best_mse = total_mse
                best_feature = col
                best_threshold = threshold
                best_left_val = left_val
                best_right_val = right_val

    # 4. Generate predictions from the best decision stump found
    if best_feature is not None:
        stump_preds = np.empty(n_samples)
        left_mask = X[:, best_feature] <= best_threshold
        stump_preds[left_mask] = best_left_val
        stump_preds[~left_mask] = best_right_val
    else:
        # Fallback to mean residual prediction if no valid split improved the score
        stump_preds = best_stump_pred

    # 5. Update the ensemble predictions using the learning rate
    updated_predictions = current_predictions + learning_rate * stump_preds

    # Round results to 4 decimal places and return as a standard Python list
    return np.round(updated_predictions, 4).tolist()