def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n_features = len(vectors)
	n_observations = len(vectors[0])
    
    
    # 1. Calculate the mean for each feature (row)
	means = []
	for entry in vectors:
		means.append(sum(entry) / len(entry))
    
    
    # 2. Initialize an n_features x n_features matrix with zeros
	covariance_matrix = [[0.] * n_features for _ in range(n_features)]
    
    
    # 3. Calculate covariance for each pair of features (i, j), which is E((X - mean(X)) * (Y - mean(Y))) = sum(X-mean(X) * Y-mean(Y)) / n_observations - 1
	for i in range(n_features):
		for j in range(n_features):
			covariance_sum = 0.0
			#go through the observations to get the sum of the products
			for k in range(n_observations):
				term_i = vectors[i][k] - means[i]
				term_j = vectors[j][k] - means[j]
				covariance_sum += term_i * term_j
			
			covariance_matrix[i][j] = covariance_sum / (n_observations - 1)

    
            
	return covariance_matrix
	