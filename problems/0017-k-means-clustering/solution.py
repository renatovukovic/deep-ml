import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here

	X = np.array(points)
	centroids = np.array(initial_centroids, dtype=float)

	for i in range(max_iterations):
		
		#compute euclidean distance first, i.e. squareroot of squared differences between the entries in the points and centroids
		distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

		#assign points to clusters based on point dimension, which is the second one
		cluster_labels = np.argmin(distances, axis=1)

		#update centroids
		new_centroids = np.zeros_like(centroids)
		for j in range(k):
			cluster_points = X[j == cluster_labels]

			#check if empty
			if len(cluster_points) > 0:
				#get new centroids
				new_centroids[j] = np.mean(cluster_points, axis=0)
			else: #keep the old centroid
				new_centroids[j] = centroids[j]

		

		#check if points are close
		if np.allclose(new_centroids, centroids):
			break
		
		centroids = new_centroids



	return [tuple(c) for c in centroids]