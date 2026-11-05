import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    # Your code here
    #formula: n over k times p^k times (1-p)^n-k
    #calculate the binomial coefficient
    binom_coeff = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    p_to_k = p**k
    one_min_k_to_nk = (1-p)**(n-k)

    #put all together
    return binom_coeff * p_to_k * one_min_k_to_nk
