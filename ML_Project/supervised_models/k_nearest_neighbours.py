import numpy as np
from utils.data_stats import euclidean_distance

class KNN():

    def __init__(self, k=5):
        self.neighbours = k

    def vote(self, neighbours_labels):
        counts = np.bincount(neighbours_labels.astype('int')) # taking the count of neighbours for most repeated class label

    def predict(self, X_test, X_train, y_train):

        predictions = np.empty(X_test.shape[0])

        for i, sample in enumerate(X_test):
            '''sorting the labels based on euclidean distance and taking only the top k'''
            idx = np.argsort([euclidean_distance(sample, point) for point in X_train])[:self.neighbours]

            k_nears = np.array([y_train[i] for i in idx])

            predictions[i] = self.vote(k_nears)

        return predictions