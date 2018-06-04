import cupy as xp 

def kernel_linear(x1, x2, params):
    return x1.dot(x2.T)

def kernel_poly(x1, x2, params):
    return (x1.dot(x2.T) + 1)**params['degree']

def kernel_rbf(x1, x2, params):
    if x2.ndim == 2:
        return xp.exp(-xp.linalg.norm(xp.subtract(x1[:, :, xp.newaxis], x2[:, :, xp.newaxis].T), axis=1)**2/params['sigma']**2)
    else:
        return xp.exp(-xp.linalg.norm(xp.subtract(x1, x2), axis=1) ** 2 / params['sigma'] ** 2)

def kernel_sigmoid(x1, x2, params):
    return xp.tanh(params['alpha'] * (x1.dot(x2.T)) + params['beta'])
    
kernel_dict = {'linear': kernel_linear,
               'poly': kernel_poly,
               'rbf': kernel_rbf,
               'sigmoid': kernel_sigmoid
                }