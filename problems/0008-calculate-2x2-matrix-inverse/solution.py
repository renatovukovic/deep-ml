def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    
    #calculate determinant to determine invertability
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    
    det = a * d - b * c

    if det == 0:
        return None

    #apply matrix formula
    coeff = (1 / det)
    A_inverse = [[coeff * d, coeff * -b], [coeff * -c, coeff * a]]

    return A_inverse
    