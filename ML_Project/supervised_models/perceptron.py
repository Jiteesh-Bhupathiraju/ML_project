import math
import numpy as np

from utils.activation_fucntion import Sigmoid
from utils.loss_fucntions import Loss

class Perceptron():
    def __init__(self, iterarions=10000, activatoin_fucntion =Sigmoid, learning_rate=0.01 , loss = Loss):
        self.iterations = iterarions
        self.learn_rate = learning_rate
        self.loss = loss()
        self.act_fn = activatoin_fucntion()

    def fit(self, X, y):
        samples, features = np.shape(X) # better to use shape as computationally efifcient than fiding the length
        _,outputs = np.shape(y)

        limit = 1/math.sqrt(features)
        self.weights = np.random.uniform(-limit, limit, (features, outputs)) # initializing the weights based on inputs and outputs
        self.bias  = np.zeros((1,outputs))

        for i in range(self.iterations):

            calculated_opt = X.dot(self.weights)+self.bias
            y_pred = self.act_fn(calculated_opt)

            # computing the error

            gradient_error= self.loss.gradient_loss(y,y_pred)*self.act_fn.gradient(calculated_opt) # calculating the slope

            grad_update_w = X.T.dot(gradient_error)
            grad_update_b = np.sum(gradient_error, axis=0, keepdims=True)

            # updating

            self.weights -= self.learn_rate*grad_update_w
            self.bias -= self.learn_rate*grad_update_b

    def predict(self,X):
        predictions = self.act_fn(X.dot(self.weights)+self.bias)