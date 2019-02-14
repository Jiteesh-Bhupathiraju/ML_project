import numpy as np
from utils.data_manipulation import divide_on_feature
from utils.data_stats import entropy

class TreeNode():
    def __init__(self, feature_id = None, threshold = None, value = None, true_branch =None, false_branch = None):
        self.feature_id = feature_id
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_bracnh = false_branch

class Decision_Tree(object):

    def __init__(self, min_samples=2, min_uncertainity=1e-7, max_depth = float('inf'), loss=None):
        self.root = None
        self.min_samples = min_samples
        self.min_uncertainity = min_uncertainity
        self.max_depth = max_depth
        self.loss = loss
        self.uncertainity_calculation = None
        self.value = None
        self.one_dim = None

    def fit(self, X, y, loss = None):
        self.one_dim = len(np.shape(y))==1
        self.root  = self.build_tree(X,y)
        self.loss = None

    def build_tree(self,X,y, depth=0):
        large_split = 0
        best_measure = None
        sets = None

        if len(np.shape(y))==1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X,y), axis=1)

        n, features = np.shape(X)

        if n>=self.min_uncertainity and depth<= self.max_depth:

            for feature in range(features):
                uniq_attributes = np.unique(np.expand_dims(X[:,feature], axis=1)) # finding the unique attributes

                for value in uniq_attributes:

                    xy1, xy2 = divide_on_feature(Xy, feature, value) # dividing on specific attribute we chose to
                                                                     # based on entropy calculation
                    if len(xy1) >0 and len(xy2) > 0:
                        y1 = xy1[:,features:]
                        y2 = xy2[:,features:]

                    uncertainity = self.uncertainity_calculation(y,y1,y2)

                    ''' getting the best feature to split the tree on '''

                    if uncertainity > large_split:
                        large_split = uncertainity
                        best_measure = {"feature_index": feature, 'attribute':value}
                        sets={'leftx':xy1[:,:features],
                             'lefty':xy1[:,features:],
                             'rightx':xy2[:,:features],
                             'righty':xy2[:,features:]}

                        ''' recursive split of the decision tree till limits are reached '''

        if large_split > self.min_uncertainity:
            true_branch = self.build_tree(sets['leftx'], sets['lefty'], depth+1)
            false_branch = self.build_tree(sets['rightx'], sets['righty'], depth+1)

            return TreeNode(feature_id=best_measure['feature_index'], threshold=best_measure['attribute'], true_branch=true_branch,
                            false_branch = false_branch)

        decision_value = self.value(y)   # decision making point

        return TreeNode(value=decision_value)



    def traverse_tree(self, x, tree=None):
        if not tree: tree= self.root

        if tree.value: return tree.value # tree refers to  a node

        attribute = x[tree.feature_id]
        branch = tree.false_branch
        if isinstance(attribute, int) or isinstance(attribute, float):
            if attribute >= tree.threshold:
                branch = tree.true_branch
        elif attribute == tree.threshold:
            branch = tree.true_branch


        return self.traverse_tree(x, branch)  # recursive traverse


    def predict(self, X):
        predictions=[self.traverse_tree(i) for i in X]
        return predictions

    def print_tree(self, tree=None, indent=' '):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("%s:%s?" %(tree.feature_id, tree.threshold) )

            print("%s->T" %(indent), end='')
            self.print_tree(tree.true_branch, indent+indent)

            print("%s->T" % (indent), end='')
            self.print_tree(tree.false_branch, indent + indent)


class classification_tree(Decision_Tree):

    def calculate_gain(self,y,y1,y2):
        p = len(y1)/len(y2)

        entrp = entropy(y)

        ''' function to calculate the gain of the specific attribute '''

        gain = entrp - p*entropy(y1)-(1-p)*entropy(y2)

        return gain


    def vote(self, y):

        """leaf value calculation form majority vote"""

        major = None
        maxi =0

        for label in np.unique(y):
            count = len(y[y==label])

            if count > maxi:
                maxi = count
                major = label

        return major

    def fit(self,X,y):

        self.uncertainity_calculation = self.calculate_gain
        self.value = self.vote

        super(classification_tree, self).fit(X,y)