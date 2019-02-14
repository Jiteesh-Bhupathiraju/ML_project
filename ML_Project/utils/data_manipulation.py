import numpy as np
import math
from itertools import combinations_with_replacement


def shuffle_data(X,y,seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0]) # shuffling the rows instead of entire data and returning
    np.random.shuffle(idx)
    return X[idx], y[idx]

def standardize(X): # setting the mean to 0 and std = 1
    X_std = X
    mean = X.mean(axis=0) # will compute the mean for all the columns and return a row of means
    std = X.std(axis=0)

    for col in range(np.shape(X)[1]):
        if std[col]:
            std[:,col] = (X_std[:,col] - mean[col])/std[col]
    return X_std

def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X,order, axis))
    l2[l2==0]=1
    return X/np.expand_dims(l2,axis)


def data_split(X,y,split_ratio=0.5, shuffle=True, seed=None):
    if shuffle:
        X,y = shuffle_data(X,y,seed)
    # finding the split index
    split_index = len(y) - int(len(y)*split_ratio)

    X_train ,X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, y_train, X_test, y_test


def convert_categorical(x,features=None):
    if not features:
        features = np.amax(x)+1 # no of uniques and 1 for the unknown
    one_hot = np.zeros((x.shape[0], features))
    one_hot[np.arange(x.shape[0], x)]=1 # setting the respective value to 1 in respective column

    return one_hot

def divide_on_feature(X, feature_index, threshold):
    split_function = None
    '''defining the function so that we can separate'''

    if isinstance(threshold, int) or isinstance(threshold, float):
        split_function = lambda sample: sample[feature_index] >= threshold
    else:
        split_function = lambda sample : sample[feature_index] == threshold

    X_with=np.array([])
    X_with_out= np.array([])

    for sample in X:
        if split_function(sample):
            np.append(X_with,sample)
        else:
            np.append(X_with_out, sample)
    return np.array([X_with, X_with_out])

def get_random_subset(X,y,subsets, replacements = True):
    samples= np.shape(X)[0]
    Xy = np.concatenate(X,y)
    np.random.shuffle(Xy)
    sets=[]

    subsample_size = samples//2

    if replacements:
        subsample_size = samples

    for _ in range(subsets):
        idx = np.random.choice(range(samples), size=np.shape(range(subsample_size)), replacement=replacements)

        X=Xy[idx][:,:-1]
        y=Xy[idx][:,-1]

        sets.append([X,y])

    return sets