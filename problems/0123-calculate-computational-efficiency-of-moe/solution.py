def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    
    n_flops_dense = n_experts * d_in * d_out
    n_flops_moe = k_active * d_in * d_out

    percentage_savings = round((n_flops_dense - n_flops_moe) / n_flops_dense * 100, 1)

    return percentage_savings