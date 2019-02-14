import numpy as np
import math

from utils.data_manipulation import data_split, normalize


class Naive_Bayes():

    def fit(self,X,y):
        self.X, self.y = X,y
        self.classes= np.unique(y)
        self.parameters=[]

        # for each and every class we save the mean and variance to calculate the gaussian probability

        for i,c in enumerate(self.classes):
            X_class = X[np.where(y==c)]
            self.parameters.append([])

            for col in X_class.T:
                param = {"mean":col.mean(), "var": col.var()}
                self.parameters[i].append(param)

    def probability(self, mean, var,x): # calculating the gaussian probability
        eps = 1e-4
        coeff = 1.0/math.sqrt(2.0*math.pi*var+eps) # so avoid 0 division when var =0
        exp = math.exp(-(math.pow(x-mean,2))/(2*var+eps))
        return coeff*exp

    def prior(self,class_label): # finding the prior probability
        rate = np.mean(self.y ==class_label)
        return rate

    def classify(self, test_example):

        post_probs=[]
        for i,c in enumerate(self.classes):
            post_prob = self.prior(c)

            for feature, param in zip(test_example, self.parameters[i]):
                prob = self.probability(param['mean'], param['var'], feature) # finding the probability based on respective feature
                post_prob*=prob                                               # mean and variance

            post_probs.append(post_prob)

        return self.classes[np.argmax(post_probs)] # picking up the maximum probability

    def predict(self,X):
        predictions = [self.classify(i) for i in X]
        return predictions