import numpy as np

# all are according to standard formulas

def linear_kernel(**kwargs):
    def result(a,b):
        return np.inner(a,b)
    return result

def poly_kernel(power,coef,**kwargs):
    def result(a,b):
        return (np.inner(a,b)+coef)**power
    return result

def rbf_kernel(gamma, **kwargs):
    def result(a,b):
        dist = np.linalg.norm(a-b)**2
        return np.exp(-gamma*dist)
    return result