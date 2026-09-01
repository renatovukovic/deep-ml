def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here

    derivative = c*n * x**(n-1)

    if n-1 < 0:
        derivative = 0
    
    return derivative

    