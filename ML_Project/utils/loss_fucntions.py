import numpy as np

from utils.activation_fucntion import Sigmoid


class Loss(object):

    def __init__(self, ):pass

    def loss(self,y,y_pred):
        return 0.5*np.power((y_pred-y),2)

    def gradient_loss(self,y,y_pred):
        return -(y-y_pred)