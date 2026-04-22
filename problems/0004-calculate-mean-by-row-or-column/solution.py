def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	result = []

	if mode == "row":
		n = len(matrix[0])
		for row in matrix:
			result.append(sum(row) / n)
	
	elif mode == "column":
		m = len(matrix)
		n = len(matrix[0])
		for i in range(n):
			column_sum = sum(matrix[j][i] for j in range(m))
			result.append(column_sum / m)

	return result