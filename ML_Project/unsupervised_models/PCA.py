import numpy as np
from utils.data_stats import covariance_matrix

class PCA():
    def project(self, X, new_dimensions):

        covairnace_matrix = covariance_matrix(X)  # finding the covariance matrix to get the eigen values
        eigen_values, eigen_vectors = np.linalg.eig(covairnace_matrix) # finding the eigens

        idx = eigen_values.argsort()[::-1] # sorting and taking the top eigens

        top_eigen_values = eigen_values[idx][:new_dimensions]
        top_eigen_vectors = np.atleast_1d(eigen_vectors[:,:idx])[:,:new_dimensions]

        X_projected = X.dot(top_eigen_vectors) # projecting on to top eigen vectors with top eigen values

        return X_projected