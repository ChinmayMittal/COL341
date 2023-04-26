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
        
class Flatten():
    def __init__(self):
        pass
    def forward(self, X):
        ## X => B * C * H * W
        ## Y => B * (C*H*W)
        self.cache = X.shape
        return X.reshape((X.shape[0], -1)) 
    def backward(self, delY):
        ## delY => B * (C*H*W) || delX => B * C * H * W 
        return delY.reshape(self.cache)
    
class MaxPool2D():
    
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, X):
        ## X ==> N * C * H * W
        (N, C_in, H_in, W_in) = X.shape
        self.cache = X
        C_out, H_out, W_out = C_in, (H_in-self.kernel_size)//self.stride + 1, (W_in-self.kernel_size)//self.stride+1
        output = np.zeros((N, C_out, H_out, W_out))
        ### iterating over the output axises
        for h in range(H_out):
            for w in range(W_out):
                ### input to consider
                h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                w_start, w_end = w * self.stride, w * self.stride + self.kernel_size
                input_slice = X[:, :, h_start:h_end, w_start:w_end] ## B * C_in * kernel_size * kernel_size
                output[:, :, h, w] = np.max(input_slice.reshape(N, C_in, -1), axis=2)
                    
        return output
    
    def backward(self, delY):
        ## delY => N * C * H_out * W_out ||| delX => N * C * H_in * W_in
        (N, C_out, H_out, W_out) = delY.shape
        output = np.zeros(self.cache.shape) ## shape of input to this layer
        ## iterating over the output
        # for channel in range(C_out):
        for h in range(H_out):
            for w in range(W_out):
                ### input to consider
                h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                w_start, w_end = w * self.stride, w * self.stride + self.kernel_size
                input_slice = self.cache[:, :, h_start:h_end, w_start:w_end] ## input which computed this output ||| B * C * K * K 
                max_value = np.max(input_slice.reshape(N, C_out, -1), axis=2) ### (B, C) max_value per batch in this window
                ## [TODO] What if multiple max values exist ?
                max_mask = ( input_slice == max_value.reshape((-1, C_out, 1, 1)))  ### B * C *  K * K ### mask indicating presence of max value
                output[:, :, h_start:h_end, w_start:w_end] = max_mask.astype(np.int32) * delY[:, :, h, w].reshape((-1, C_out, 1, 1))  ### B * C *  K * K
        return output
    
class Conv2D():
    
    def __init__(self, kernel_size, stride, in_channels, num_filters):
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.W = np.random.uniform(low=-0.1, high=+0.1, size=(num_filters, in_channels, kernel_size, kernel_size)) ### C_out * C_in * K * K
        self.b = np.zeros(shape=(num_filters,)) ## C_out, ||| one bias per output filter


    def forward(self, X):
        ### X => N * C * H * W
        self.cache = X
        (N, C_in, H_in, W_in) = X.shape
        C_out, H_out, W_out = self.num_filters, (H_in-self.kernel_size)//self.stride + 1, (W_in-self.kernel_size)//self.stride+1
        output = np.zeros(shape=(N, C_out, H_out, W_out))
        ### iterating over the output
        for h in range(H_out):
            for w in range(W_out):
                ### input to consider
                h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                w_start, w_end = w * self.stride, w * self.stride + self.kernel_size
                input_slice = X[:, :, h_start:h_end, w_start:w_end] ## B * C_in * K * K
                                    ### C_out * C_in * K * K              B * 1 * C_in * K * K => B * C_out * C_in * K * K 
                output[:, :, h, w] = np.sum( (self.W[:, :, :, :] * np.expand_dims(input_slice, axis=1)).reshape(N, C_out, -1), axis=2) + self.b ## B, C_out
                    
        return output
    
    
    def backward(self, delY):
        ### delY => N * C_out *  H_out * W_out
        (N, C_out, H_out, W_out) = delY.shape
        output = np.zeros(self.cache.shape) ### delX
                        ### shift the C_out to the zeroth axis 
        
        self.b_grad = np.sum(np.moveaxis(delY, source=1, destination=0).reshape(C_out, -1), axis=1) ### for each channel sum all the gradient values
        
        self.W_grad = np.zeros(shape=self.W.shape) ### C_out * C_in * K * K
        for h in range(H_out):
            for w in range(W_out):
                ### input to consider
                h_start, h_end = h * self.stride, h * self.stride + self.kernel_size
                w_start, w_end = w * self.stride, w * self.stride + self.kernel_size
                input_slice = self.cache[:, :, h_start:h_end, w_start:w_end] ## B * C_in * K * K
                ## C_out * C_in * K * K            B * _ * C_in * K * K  ||| B, C_out , _1_, _1_, _1_
                self.W_grad[:, :, :, :] += np.sum(np.expand_dims(input_slice, axis=1) * (delY[:, :, h, w].reshape(N, C_out, 1, 1, 1)), axis=0)
                ## B * C_in * K * K                               C_out * C_in * K * K  ||| B, C_out, 1, 1  ==>  _1_ * C_out * C_in * K * K  || B * C_out * _1_ * 1 * 1 => then sum across outputs channels
                output[:, :, h_start:h_end, w_start:w_end] += np.sum(np.expand_dims(self.W[:, :, :, :], axis=0) * delY[:, :, h, w].reshape(N, C_out, 1, 1, 1), axis=1)
    
        return output
    
    def update(self, lr):
        self.W -= lr * self.W_grad
        self.b -= lr * self.b_grad
                    
            