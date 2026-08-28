import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here

    intermediate_objective_list = []
    for i in range(len(rhos)):
        rho_times_advantage = rhos[i] * A[i]
        rho_clipped = np.clip(rhos[i], 1-epsilon, 1+epsilon)
        rho_clipped_times_advantage = rho_clipped * A[i]

        min_rho = min(rho_times_advantage, rho_clipped_times_advantage)

        #subtract KL penalty
        #kl penalty from description
        kl_penalty = rhos[i] * ((pi_theta_ref[i] / (rhos[i] * pi_theta_old[i])) - np.log(pi_theta_ref[i] / (rhos[i] * pi_theta_old[i])) - 1)
        #exact kl penalty
        #kl_penalty = np.log(pi_theta_old[i] / pi_theta_ref[i])

        final_value = min_rho - beta * kl_penalty

        intermediate_objective_list.append(final_value)

    #the mean is the final result
    result = np.mean(intermediate_objective_list)

    return result
    