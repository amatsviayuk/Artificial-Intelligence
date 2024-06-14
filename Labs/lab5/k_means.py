import numpy as np


def initialize_centroids_forge(data, k):
    n_samples = data.shape[0]  # rows
    random_indices = np.random.choice(n_samples, k, replace=False)
    return data[random_indices]


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point sqrt(sum(data - centroids)^2)
    distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)  # getting the closest centroid index


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    unique_clusters = np.unique(assignments)
    centroids = np.zeros((len(unique_clusters), data.shape[1]))

    for i, cluster in enumerate(unique_clusters):
        cluster_data = data[assignments == cluster]  # getting cluster's data points
        centroid = np.mean(cluster_data, axis=0)  # avg
        centroids[i] = centroid
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids):
    # centroids initialization
    centroids = initialize_centroids_forge(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)

