import  numpy as np
import math

from utils.data_manipulation import divide_on_feature, get_random_subset
from supervised_models.decision_tree import classification_tree

class random_forest():

    def __init__(self, estimators=100, features_limit=None, min_split=2, min_gain=0, max_depth = float('inf')):

        self.estimators = estimators
        self.feature_limit = features_limit
        self.min_split = min_split
        self.min_gain = min_gain
        self.max_depth = max_depth


        self.trees=[]

        ''' n classification trees in the forest '''

        for _ in range(estimators):
            self.trees.append(classification_tree(min_samples=self.min_split, min_uncertainity=min_gain, max_depth=self.max_depth))




    def fit(self, X, y):

        total_features = np.shape(X)[1]
        if not self.feature_limit:
            ''' setting the limit for the features '''
            self.feature_limit = int(math.sqrt(total_features))

        ''' n random sets for n classifiers '''

        subsets = get_random_subset(X,y,self.estimators)

        ''' training each tree in the forest '''

        for i in range(self.estimators):
            Xsub, ysub = subsets[i]

            idx = np.random.choice(range(total_features), size=self.feature_limit, replace=True)

            self.trees[i].feature_indices = idx
            Xsub = Xsub[:,idx]

            self.trees[i].fit(Xsub,ysub)


    def predict(self, X):
        predictions=np.empty((X.shape()[0], len(self.trees)))

        for i, tree in  enumerate(self.trees):
            ''' for traversing through the tree, for all the samples using a single tree, for making a prediction '''
            idx = tree.feature_indices

            pred = tree.predict(X[:,idx])

            predictions[:,i] = pred


        prediction = []

        for p in predictions:
            ''' taking the majority count '''
            prediction.append(np.bincount(p.astype('int')).argmax())

        return prediction