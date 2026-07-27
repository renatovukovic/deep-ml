import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    
    #for softmax divide the exp of the values by the sum of the exps, subtract max for numerical stability

    max_val = max(scores)
    for i in range(len(scores)):
        current = math.exp(scores[i] - max_val)
        scores[i] = current

    sum_val = sum(scores)

    for i in range(len(scores)):
        scores[i] /= sum_val

    return scores