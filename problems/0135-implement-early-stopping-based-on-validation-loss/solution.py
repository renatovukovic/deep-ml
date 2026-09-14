from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    no_improvement_count = 0
    best_epoch = 0
    for i in range(1, len(val_losses)):
        current_difference = val_losses[best_epoch] - val_losses[i]
        if current_difference > min_delta:
            best_epoch = i
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience: #stop early
            break

    
    return min(best_epoch + patience, len(val_losses)-1), best_epoch
