import numpy as np
from utils.data_manipulation import normalize
from utils.data_stats import euclidean_distance

class k_means_clustering():
    def __init__(self,k=2, iterations=500):

        self.k = k
        self.iter_limit=iterations

    def pick_random_centroids(self,X): # picking random centroids from the data itself
        n, features = np.shape(X)
        centroids = np.zeros((self.k, features))

        for i in range(self.k):
            center = X[np.random.choice(range(n))] # choosing random sample of the input
            centroids[i]=center

        return centroids

    def nearest_center(self, point, centers):
        close_i =0
        min_dist = float('inf')

        for i, c in enumerate(centers):
            d = euclidean_distance(point, c) # using the euclidean metric to find the closest point
            if d < min_dist:
                close_i = i
                min_dist = d
        return close_i

    def recreate_clusters(self, centers, X):
        n = np.shape(X)[0]
        clusters=[[] for _ in range(self.k)]

        for i, point in enumerate(X):
            closer = self.nearest_center(point, centers)
            clusters[closer].append(point)    # recreating the clusters

        return clusters

    def recalculate_centroids(self, clusters, X):
        features=np.shape(X)[1]
        new_centroids = np.zeros((self.k, features))

        for i,cluster in enumerate(clusters):
            new_center = np.mean(X[cluster], axis=0)   # finding the new centers by taking the mean of the new labeled points
            new_centroids[i] = new_center

        return new_centroids


    def cluster_labels(self, clusters, X):

        labels=np.zeros(np.shape(X)[0])
        for i, cluster in enumerate(clusters):
            for point in cluster:
                labels[point] = i # assigning the cluster label after the algorithm converges

        return labels



    def predict(self, X):

        centroids = self.pick_random_centroids(X)

        for _ in range(self.iter_limit):
            clusters = self.recreate_clusters(centroids, X)
            prev_centroids  = centroids
            centroids = self.recalculate_centroids(clusters, X)

            diff = centroids - prev_centroids

            if not diff.any():
                break

        return self.cluster_labels(clusters, X)