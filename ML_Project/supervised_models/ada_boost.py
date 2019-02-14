import numpy as np
import math
from sklearn import datasets
from utils.data_stats import accuracy_score

class weak_classifier():

    def __init__(self):
        self.class_label=1
        self.feature_index=None   # for making prediction
        self.threshold = None     # of a certain threshold
        self.weight=None



class Adaboost():
    def __init__(self, classifiers_count=5):
        self.classifiers_count = classifiers_count

    def fit(self, X,y):

        samples, features = np.shape(X)
        W = np.full(samples, (1/samples))   # initializing weights for all the samples equally

        self.classifiers=[]

        for _ in range(self.classifiers_count):
            classifier=weak_classifier()
            min_error = float('inf')

            for feature in range(features):
                feature_values = np.expand_dims(X[:,feature], axis=1)
                uniq_attributes = np.unique(feature_values)

                for attribute in uniq_attributes:
                    p=1

                    prediction = np.ones(np.shape(y))
                    ''' making predictions on an attribute for a specific feature '''
                    prediction[X[:, feature]<attribute] = -1
                    error = sum(W[y!=prediction])


                    if error>0.5:
                        error = 1-error
                        p-=1
                    ''' picking the right attribute in a feature to classify  '''

                    if error < min_error:
                        classifier.class_label = p
                        classifier.threshold = attribute
                        classifier.feature_index = feature
                        min_error = error

            ''' updating a classifier weight based on its accuracy '''
            classifier.weight = 0.5*math.log((1.0-min_error)/(min_error+1e-10))
            predictions = np.ones(np.shape(y))
            negative_idx = (classifier.class_label*X[:,classifier.feature_index]<classifier.class_label*classifier.threshold)

            ''' updating the predictions '''
            predictions[negative_idx]=-1
            W*=np.exp(-classifier.weight*y*predictions)

            ''' and so as weights '''
            W/=np.sum(W)

            self.classifiers.append(classifier)


    def predict(self, X):
        samples =  np.shape(X)[0]
        predictions=np.zeros((samples,1))

        for classifier in self.classifiers:
            preds=np.ones(np.shape(predictions))
            negative_idx = (classifier.class_label * X[:, classifier.feature_index] < classifier.class_label * classifier.threshold)
            preds[negative_idx]=-1

            ''' making the predictions based on the weights
                updating the prediction of a point from a specific classifier'''

            predictions+=classifier.weight*preds

        predictions=np.sign(predictions).flatten()

        return predictions