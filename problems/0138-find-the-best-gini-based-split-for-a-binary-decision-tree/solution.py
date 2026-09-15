import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_feature_index = 0
    best_threshold = 0.0
    
    # Get the total counts of each class in the parent node
    classes, parent_counts = np.unique(y, return_counts=True)
    
    # 1. Loop through each feature (column)
    for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        
        # 2. Sort the feature values and rearrange y accordingly
        sort_idx = np.argsort(X_column)
        X_sort, y_sort = X_column[sort_idx], y[sort_idx]
        
        # Keep a running tally of class counts for the left split
        # This avoids expensive re-calculation at every single threshold
        left_counts = np.zeros_like(parent_counts)
        
        # 3. Evaluate candidate splits between adjacent samples
        for i in range(n_samples - 1):
            # Find which class index the current sample belongs to
            class_idx = np.where(classes == y_sort[i])[0][0]
            left_counts[class_idx] += 1
            
            # Skip duplicate feature values (a split is only possible between different values)
            if X_sort[i] == X_sort[i + 1]:
                continue
                
            # Right split counts are just total counts minus left split counts
            right_counts = parent_counts - left_counts
            
            # Number of samples in each child
            n_left = i + 1
            n_right = n_samples - n_left
            
            # 4. Calculate Gini Impurity for Left and Right children
            gini_left = 1.0 - np.sum((left_counts / n_left) ** 2)
            gini_right = 1.0 - np.sum((right_counts / n_right) ** 2)
            
            # 5. Compute Weighted Gini Impurity
            weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
            
            # 6. Update best split if this one is better
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_idx
                # Threshold is the midpoint between adjacent unique values
                best_threshold = X_sort[i] # + X_sort[i + 1]) / 2.0
                
    return best_feature_index, best_threshold

