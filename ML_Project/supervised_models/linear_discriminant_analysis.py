import numpy as np
from utils.data_stats import correlation_matrix, covariance_matrix

class LDA():
    def __init__(self):
        self.vector = None

    def transform(self,X,y):
        self.fit(X,y)

    def fit(self,X,y):

        A = X[y==0] # separation of classes
        B = X[y==1]

        covA = covariance_matrix(A) # covariances
        covB = covariance_matrix(B)
        combine_cov = covA + covB

        meanA = A.mean(axis=0) # means
        meanB = B.mean(axis=0)
        mean_AB = np.atleast_1d(meanA - meanB)

        # finding the best covriate vector
        self.vector = np.linalg.pinv(combine_cov).dot(mean_AB)

    def project(self,X):
        predictions=[]
        for sample in X:
            p = sample.dot(self.vector) # projecting each example onto the
            p = 1*(p<0)
            predictions.append(p)

        return predictions # returnning the resulting vector