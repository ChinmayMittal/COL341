import numpy as np
import sklearn.linear_model as lm


class LinearRegression():
    
    
    def __init__(self, loss="MSE", num_features=None, learning_rate=0.001, stopping_criterion="maxit", max_iters=10, val_loss_decrease_threshold = .01, lambda_ = 5):
        """_summary_

        Args:
            loss (str, optional): used for computing gradients. Defaults to "MSE". | options "ridge" | "MSE"
            num_features (_type_, optional): number of features in the model. Defaults to None.
            learning_rate (float, optional): learning rate for gradient descent. Defaults to 0.001.
            stopping_criterion (str, optional): stopping criterion using validation loss. Defaults to "maxit". | options "reltol" and "maxit"
            max_iters (int, optional): max iterations for stopping_criterion => "maxit". Defaults to 10.
            val_loss_decrease_threshold (float, optional): val loss decrease threshold for stopping_criterion => "reltol" . Defaults to .01.
            lambda_ : float : regularization parameter for ridge regression 
        """
        self.error_metric = loss ### MSE or Ridge Loss
        self.num_features = num_features ### without the bias term
        self.w = np.zeros((self.num_features+1,), dtype=np.float64) ### first term is bias
        self.learning_rate = learning_rate
        self.stopping_criterion = stopping_criterion
        self.max_iters = max_iters
        self.train_loss = []
        self.val_loss = []
        self.num_updates = 0
        self.val_loss_decrease_threshold = val_loss_decrease_threshold
        self.lambda_ = lambda_
        
    def add_bias_column(self, X):
        ### add the bias column of all ones to the X (as first column)
        ### transform it form N*d => N*d+1 
        ones = np.ones((len(X), 1), dtype=np.float64) #### N * 1
        one_X = np.concatenate((ones, X), axis=1)
        return one_X
    
    def custom_loss(self, X, y, loss = "MSE", bias_added=False):
        #### X is N*d, y => N
        if not bias_added:
            X = self.add_bias_column(X)
        pred = np.dot(X, self.w)
        error = pred - y
        if loss == "MSE":
            return np.mean(error**2)
        elif loss == "MAE":
            return np.mean(np.abs(error))
        
    def loss(self, X, y):
        ### X is N * d+1, y => N 
        pred = np.dot(X, self.w)
        error = pred  - y
        if self.error_metric == "MSE":
            error = np.sum(error**2)/(2*len(X))
        elif self.error_metric == "ridge":
            error = np.sum(error**2)/(2*(len(X))) + self.lambda_ * (np.abs(self.w**2) - self.w[0]**2)
    
        return error
    
    def gradient(self, X, y):
        ### X => N * d+1, y => N
        pred = np.dot(X, self.w )
        error = (pred - y) 
        error = np.reshape(error, newshape=(-1, 1))
        grad_matrix = X * error
        if self.error_metric == "MSE":
            grad = np.mean(grad_matrix, axis=0)
        elif self.error_metric == "ridge":
            regularizing_term = 2 * self.lambda_ * self.w
            regularizing_term[0] = 0 ### don't regularize bias
            grad = np.mean(grad_matrix, axis=0) + regularizing_term
        
        return grad
            
    
    def training_finished(self):
        
        if self.stopping_criterion == "maxit":
            return self.num_updates >= self.max_iters
        elif self.stopping_criterion == "reltol":
            if(len(self.val_loss) < 2):
                return False
            else:
                val_loss_decrease = self.val_loss[-1]-self.val_loss[-2]
                relative_decrease = val_loss_decrease / (self.val_loss[-2])
                return -1*relative_decrease < self.val_loss_decrease_threshold        
    
        
    def update_weights(self, train_X, train_y, val_X, val_y):

        train_X = self.add_bias_column(train_X)
        val_X = self.add_bias_column(val_X)

        if(self.num_updates == 0):
            self.train_loss.append(self.custom_loss(train_X, train_y, loss="MSE", bias_added=True))
            self.val_loss.append(self.custom_loss(val_X, val_y, loss="MSE", bias_added=True))

        gradient_t = self.gradient(train_X, train_y)
        self.w = self.w - self.learning_rate * gradient_t

        self.num_updates += 1
        
        self.train_loss.append(self.custom_loss(train_X, train_y, loss="MSE", bias_added=True))
        self.val_loss.append(self.custom_loss(val_X, val_y, loss="MSE", bias_added=True))
        
    def pred(self, X):

        ## X is N * d
        X = self.add_bias_column(X)
        pred = np.dot(X, self.w)
        return pred
        
        
class ScikitLearnLR():
    
    
    def __init__(self):
        self.model = lm.LinearRegression(fit_intercept=True, n_jobs=-1)
        self.train_loss = None
        self.val_loss = None
        
    def add_bias_column(self, X):
        ### add the bias column of all ones to the X (as first column)
        ### transform it form N*d => N*d+1 
        ones = np.ones((len(X), 1), dtype=np.float64) #### N * 1
        one_X = np.concatenate((ones, X), axis=1)
        return one_X
    
    def custom_loss(self, X, y, loss = "MSE", bias_added=False):
        #### X is N*d, y => N
        pred = self.model.predict(X)
        error = pred - y
        if loss == "MSE":
            return np.mean(error**2)
        elif loss == "MAE":
            return np.mean(np.abs(error))
        
    def update_weights(self, train_X, train_y, val_X, val_y):
        self.model.fit(train_X, train_y)
    
    
    def training_finished(self):
        return True ### only one pass for scikit learn
    
    def pred(self, X):
        ## X is n*d
        return self.model.predict(X)