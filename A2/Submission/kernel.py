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
    
    # computes the general polynomial kernel => (<x1,x2> + c)^d
    # print(kwargs)
    c, d = kwargs["poly_c"], kwargs["poly_d"]
    assert X.ndim == 2
    K = linear(X, **kwargs)
    # return K ** 0 + K + 0.01 * K**2
    return (K+c)**d
    
def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ### exp(-gamma <x - x' , x - x'>)
    gamma = kwargs["rbf_gamma"]
    K = linear(X, **kwargs)
    
    dot_prod_mat = np.sum( X ** 2, axis = 1 )
    dot_prod_mat = np.reshape(dot_prod_mat, newshape=(X.shape[0],1))
    dot_prod_mat = np.hstack([dot_prod_mat]*X.shape[0])
    
    ans = np.exp(-gamma * (dot_prod_mat + dot_prod_mat.T - 2*K ))
    
    return ans
    
def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ### tanh( gamma  * < x, x' > + r )
    gamma, r = kwargs["sigmoid_gamma"], kwargs["sigmoid_r"]
    K = linear(X,**kwargs)
    return np.tanh(gamma* K + r)

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    
    ## exp(-gamma * || x - x' ||_1 )
    gamma = kwargs["laplacian_gamma"]
    
    n = X.shape[0]
    K = np.zeros(shape=(n,n))
    
    for i in range(n):
        for j in range(n):
            K[i][j] = np.exp(-gamma * np.sum(np.abs(X[i,:]-X[j,:])))


    return K

## helper function to compute the kernel value of x with all vectors in X
def linear_(x, X, **kwargs):
    ### returns a np array of the kernel value of x with all vectors in X
    ## x => 1 * N_features
    return np.dot(x, X.T)

def polynomial_(x,X,**kwargs):
    c, d = kwargs["poly_c"], kwargs["poly_d"]
    K = linear_(x,X,**kwargs)
    return (K+c)**d

def rbf_(x, X, **kwargs):
    gamma = kwargs["rbf_gamma"]
    K = linear_(x,X,**kwargs)
    dot_prod = np.sum(X**2, axis=1, keepdims=True).T
    return np.exp(-gamma*( np.sum(x*x) + dot_prod - 2*K))

def sigmoid_(x,X,**kwargs):
    ### tanh( gamma  * < x, x' > + r )
    gamma, r = kwargs["sigmoid_gamma"], kwargs["sigmoid_r"]
    K = linear_(x,X,**kwargs)
    return np.tanh(gamma* K + r)

def laplacian_(x,X,**kwargs):
    ## exp(-gamma * || x - x' ||_1 )
    gamma = kwargs["laplacian_gamma"]
    n = X.shape[0]
    K = np.zeros(shape=(1,n))
    
    for i in range(n):
            K[0][i] = np.exp(-gamma * np.sum(np.abs(X[i,:]-x)))


    return K

#### helper function to invoke the correct kernel function
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

def get_k_mat_(x, X, kernel, **kwargs):
    
    if kernel == "linear":
        K_mat = linear_(x, X, **kwargs)
    elif kernel == "polynomial":
        K_mat = polynomial_(x, X, **kwargs)
    elif kernel == "rbf":
        K_mat = rbf_(x, X, **kwargs)
    elif kernel == "sigmoid":
        K_mat = sigmoid_(x, X, **kwargs)
    elif kernel == "laplacian":
        K_mat = laplacian_(x, X, **kwargs)
        
    return K_mat

