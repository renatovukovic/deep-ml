def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:

	#equation for eigenvalues of matrix A is det(A - lambda * I) = 0
	# det(A) = ad - bc =(a-lambda) * (d-lambda) - bc
	# = ad - (a+d)*lambda + lambda^2 - bc 
	# = lambda^2 - (a+d)*lambda + ad - bc
	# <=> lambda * (lambda - (a+d)) = -ad + bc
	# lambda = - (-(a+d) / 2) +- sqrt((-(a+d)/2)^2 - (ad - bc))

	a, b = matrix[0]
	c, d = matrix[1]

	part1_pq = -(-(a+d) / 2)
	sqrt_pq = ((-(a+d) / 2)**2 - (a*d - b*c)) ** (1/2)

	lambda1 = part1_pq + sqrt_pq
	lambda2 = part1_pq - sqrt_pq


	#eigenvalues = [lambda1, lambda2]	

	eigenvalues = sorted([lambda1, lambda2], reverse=True)

	return eigenvalues