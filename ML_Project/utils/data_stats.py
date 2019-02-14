import  numpy as np
import math

def entropy(y): #array
    log2 = lambda x: math.log(x)//math.log(2)
    uniq = np.unique(y)
    entropy_value = 0
    for label in uniq:
        count = len(y[y==label])
        percent = count/len(y)
        entropy_value+= -percent*log2(percent) # we are summing the split of the labels in a given array
    return entropy_value

def mean_sqaure_error(y_true, y_pred):
    mse = np.mean(np.power(y_true, y_pred,2))
    return mse

def features_variance(X):
    mean = np.ones(np.shape(X))*X.mean(0)
    total_n = np.shape(X)[0]
    feat_var = (1/total_n)*np.diag((X-mean).T.dot(X-mean))

    return feat_var

def feature_std(X):
    feat_std = np.sqrt(features_variance(X))
    return feat_std


def euclidean_distance(x,y):
    distance=0
    for i in range(len(x)):
        distance += pow((x[i]-y[i]),2)

    return math.sqrt(distance)


# based on formulas
def covariance_matrix(X, Y=None):
    if not Y:
        Y=X
    total_n = np.shape(X)[0]
    cv = (1/total_n-1)*(X-X.mean(axis=0)).T.dot(Y- Y.mean(axis=0))

    return np.array(cv,dtype=float)

def correlation_matrix(X,Y=None):
    if not Y:
        Y=X
    total_n = np.shape(X)[0]
    cv = (1/total_n)*(X-X.mean(axis=0)).T.dot(Y- Y.mean(axis=0))
    std_x = np.expand_dims(feature_std(X),1)
    std_y = np.expand_dims(feature_std(Y),1)
    cm = np.divide(cv,std_x.dot(std_y.T))

    return np.array(cm,dtype=float)


def accuracy_score(true_values, predicted_values):
    accuracy = np.sum(true_values==predicted_values, axis=0)/len(true_values)
    return accuracy