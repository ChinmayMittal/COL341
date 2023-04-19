import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features, self.out_features = in_features, out_features
        self.W = np.random.uniform(low=-0.1, high=+0.1, size=(in_features, out_features))
        self.b = np.zeros(shape=(out_features,))
    def forward(self, X):
        ### X is M * D_in
        ## X.W + b ===> X (M * D_IN ) . W (D_IN * D_OUT) + b (D_out)
        self.cache_X = X
        return np.matmul(X, self.W) + self.b

    def backward(self, delY):
        ### delY => M * D_out
        ### each term is the gradient of the scalar loss wrt the corresponding element in the input
        self.W_grad = np.matmul(self.cache_X.T, delY)
        self.b_grad = np.sum(delY, axis=0)
        delX = np.matmul(delY, self.W.T)
        return delX
    
    def update(self, lr):
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad
    
class ReLU:
    def __init_(self):
        pass
    
    def forward(self, X):
        ## Y = ReLU(X)
        self.X_cache = X
        return np.maximum(X, 0)
    
    def backward(self, delY):
        return np.where(self.X_cache >= 0, 1, 0) * delY
    
class SoftMax:
    def __init__(self):
        pass
    
    def forward(self, X):
        ## Y = softmax(X) || X => M * NUM_CLASSES
        self.X_cache = X
        denominator = np.sum(np.exp(X), axis=1, keepdims=True) ## M, 1
        self.prob_X_cache =  np.exp(X) / denominator
        return self.prob_X_cache
    
    def backward(self, delY):
        ### Y is M * NUM_CLASSES
        self.ans = np.zeros(shape=(self.X_cache.shape))
        for sample_idx in range(self.X_cache.shape[0]):
            prob = self.prob_X_cache[sample_idx] ### probabilities for this data point ||| (NUM_CLASSES, )
            A = np.diag(prob) - np.matmul(prob.reshape((-1,1)), prob.reshape(1, -1))
            self.ans[sample_idx] =  np.matmul(delY[sample_idx, :].reshape(1, -1), A).reshape(-1)
        return self.ans
        
