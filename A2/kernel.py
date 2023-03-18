import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    assert X.ndim == 2
    kernel_matrix = X @ X.T
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    pass

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    pass

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    pass

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    pass

