import numpy as np
import cvxopt
from utils.misc import *


class SVM(object): # for binary classification

    def __init__(self, penality=1, kernel=rbf_kernel, power=4, gamma=None, coeff=4):

        self.penality=penality
        self.power = power
        self.gamma = gamma
        self.coeff= coeff

        self.lagrang_multipliers=None
        self.support_vectors = None
        self.support_vectors_labels = None
        self.intercept = None

    def fit(self, X,y):
        n, features = np.shape(X)

        if not self.gamma:
            self.gamma = 1/features

        self.kernel = self.kernel(power = self.power, coeff = self.coeff, gamma = self.gamma )

        kernel_matrix  = np.zeros((n,n))
        '''calculating the kernel matrix'''
        for i in range(n):
            for j in range(n):
                kernel_matrix = self.kernel(X[i], X[j])

        '''Quadratic optimization problem'''

        P = cvxopt.matrix(np.outer(y,y)*kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n)*-1)
        A = cvxopt.matrix(y,(1,n), tc='d')
        b = cvxopt.matrix(0,tc='d')

        '''penality consideraiton'''

        if not self.penality:
            G = cvxopt.matrix(np.identity(n)*-1)
            h = cvxopt.matrix(np.zeros(n))

        else:
            G_max = np.identity(n)*-1
            G_min = np.identity(n)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))

            h_max = cvxopt.matrix(np.zeros(n))
            h_min =cvxopt.matrix(np.ones(n), self.penality)

            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        ''' solution for quadratic equation '''

        minimize = cvxopt.solvers.qp(P,q,G,h,A,b)

        lagrange_multipliers = np.ravel(minimize['x'])

        '''considering only the non zero which can stay as the support vectors'''

        idx = lagrange_multipliers > 1e-7

        self.lagrang_multipliers = lagrange_multipliers[idx]

        '''support vectors and the labels'''
        self.support_vectors = X[idx]
        self.support_vectors_labels = y[idx]

        self.intercept = self.support_vectors_labels[0]

        '''intercept'''

        for i in range(len(self.lagrang_multipliers)):
            self.intercept -= self.lagrang_multipliers[i]*self.support_vectors_labels[i]*self.kernel(self.support_vectors[i], self.support_vectors[0])



    def predict(self,X):

        predictions=[]

        for sample in X:
            pred =0
            '''calculating the prediction value'''
            for i in range(len(self.lagrang_multipliers)):
                pred+=self.lagrang_multipliers[i]*self.support_vectors_labels[i]*self.kernel(self.support_vectors[i], sample)

            pred+=self.intercept
            '''assigning the sign'''
            predictions.append(np.sign(pred))

        return np.array(predictions)