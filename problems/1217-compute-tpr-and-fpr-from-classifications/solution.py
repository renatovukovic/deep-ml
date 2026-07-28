import numpy as np

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    # TODO: compute TP, FN, FP, TN then TPR and FPR
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] == 0:
                tn += 1
            else:
                fp += 1

    
    #calculate the rates now
    if tp + fn: #zero division
        tpr = tp / (tp + fn)
    else:
        tpr = 0
    if fp + tn:
        fpr = fp / (fp + tn)
    else:
        fpr = 0

    return (tpr, fpr)
