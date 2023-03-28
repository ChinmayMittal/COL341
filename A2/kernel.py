import numpy as np
from utilities import nCr

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
    
    # computes the general polynomial kernel => (<x1,x2> + c)^d
    c, d = 1, 2
    
    assert X.ndim == 2
    K = linear(X, **kwargs)
    # return K ** 0 + K + 0.01 * K**2
    ans = np.zeros(shape=K.shape)
    for i in range(0,d+1):
        ans += (nCr(d,i) * (c**i) ) * ( K ** (d-i)) 
        
    return ans
    
    
    

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ### exp(-gamma <x - x' , x - x'>)
    gamma = .1
    K = linear(X, **kwargs)
    
    dot_prod_mat = np.sum( X ** 2, axis = 1 )
    dot_prod_mat = np.reshape(dot_prod_mat, newshape=(X.shape[0],1))
    dot_prod_mat = np.hstack([dot_prod_mat]*X.shape[0])
    
    ans = np.exp(-gamma * (dot_prod_mat + dot_prod_mat.T - 2*K ))
    
    return ans
    
    
    

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ### tanh( gamma  * < x, x' > + r )
    gamma, r = 1,1
    K = linear(X,**kwargs)
    return np.tanh(gamma* K + r)

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ## exp(-gamma * || x - x' ||_1 )
    gamma = 0.001
    
    n = X.shape[0]
    K = np.zeros(shape=(n,n))
    
    for i in range(n):
        for j in range(n):
            K[i][j] = np.exp(-gamma * np.sum(np.abs(X[i,:]-X[j,:])))


    return K

def get_k_mat(X, kernel, **kwargs):
    
    if kernel == "linear":
        K_mat = linear(X, **kwargs)
    elif kernel == "polynomial":
        K_mat = polynomial(X, **kwargs)
    elif kernel == "rbf":
        K_mat = rbf(X, **kwargs)
    elif kernel == "sigmoid":
        K_mat = sigmoid(X, **kwargs)
    elif kernel == "laplacian":
        K_mat = laplacian(X, **kwargs)
        
    return K_mat