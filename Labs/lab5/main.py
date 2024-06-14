from k_means import k_means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    # print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes


def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters == cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster == label_type)}")


def plot_data_and_centroids(features, assignments, centroids):
    # Data points
    plt.scatter(features[:, 0], features[:, 1], c=assignments)
    # Centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

    plt.title('Data points and centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def clustering():
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    for i in range(20):
        assignments, centroids, error = k_means(features, 3)

        # evaluate(assignments, classes)
        intra_class_variance.append(error)
    plot_data_and_centroids(features, assignments, centroids)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")


if __name__ == "__main__":
    print("Forgy")
    clustering()
