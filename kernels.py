import cupy as xp 
from sklearn.metrics.pairwise import rbf_kernel

def kernel_linear(x1, x2, params):
    return x1.dot(x2.T)

def kernel_poly(x1, x2, params):
    return (x1.dot(x2.T) + 1)**params['degree']

def kernel_rbf(x1, x2, params):
    if x2.ndim == 2:
        return xp.exp(-xp.linalg.norm(xp.subtract(x1[:, :, xp.newaxis], x2[:, :, xp.newaxis].T), axis=1)**2/params['sigma']**2)
    else:
        return xp.exp(-xp.linalg.norm(xp.subtract(x1, x2), axis=1) ** 2 / params['sigma'] ** 2)

def kernel_rbf_sklearn(x1, x2, params):
    return xp.asarray(rbf_kernel(xp.asnumpy(x1), gamma=params['sigma']))

def kernel_sigmoid(x1, x2, params):
    return xp.tanh(params['alpha'] * (x1.dot(x2.T)) + params['beta'])
    
kernel_dict = {'linear': kernel_linear,
               'poly': kernel_poly,
               'rbf': kernel_rbf,
               'rbf_sklearn': kernel_rbf_sklearn,
               'sigmoid': kernel_sigmoid
                }