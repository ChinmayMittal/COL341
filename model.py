import numpy as np



class LinearRegression():
    
    
    def __init__(self, loss="MSE", num_features=None, learning_rate=0.001, stopping_criterion="maxit", max_iters=10):
        
        self.error_metric = loss ### MSE or Ridge Loss
        self.num_features = num_features ### without the bias term
        self.w = np.zeros((self.num_features+1,), dtype=np.float64) ### first term is bias
        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion
        self.max_iters = max_iters
        self.train_loss = []
        self.val_loss = []
        self.num_updates = 0
        
    def add_bias_column(self, X):
        ### add the bias column of all ones to the X (as first column)
        ### transform it form N*d => N*d+1 
        ones = np.ones((len(X), 1), dtype=np.float64) #### N * 1
        one_X = np.concatenate((ones, X), axis=1)
        return one_X
        
    def loss(self, X, y):
        ### X is N * d+1, y => N 
        pred = np.dot(X, self.w)
        error = pred  - y
        if self.error_metric == "MSE":
            error = np.sum(error**2)/(2*len(X))
    
        return error
    
    def gradient(self, X, y):
        ### X => N * d+1, y => N
        pred = np.dot(X, self.w )
        if self.error_metric == "MSE":
            error = (pred - y) 
            error = np.reshape(error, newshape=(-1, 1))
            grad_matrix = X * error
            grad = np.mean(grad_matrix, axis=0)
        
        return grad
            
    
    def training_finished(self):
        
        if self.stopping_criterion == "maxit":
            return self.num_updates >= self.max_iters
    
        
    def update_weights(self, train_X, train_y, val_X, val_y):

        train_X = self.add_bias_column(train_X)
        val_X = self.add_bias_column(val_X)

        if(self.num_updates == 0):
            self.train_loss.append(self.loss(train_X, train_y))
            self.val_loss.append(self.loss(val_X, val_y))

        gradient_t = self.gradient(train_X, train_y)
        self.w = self.w - self.learning_rate * gradient_t

        self.num_updates += 1
        
        self.train_loss.append(self.loss(train_X, train_y))
        self.val_loss.append(self.loss(val_X, val_y))
        
    def pred(self, X):

        ## X is N * d
        X = self.add_bias_column(X)
        pred = np.dot(X, self.w)
        return pred
        
        
        
    
