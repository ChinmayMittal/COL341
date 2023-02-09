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
            return self.num_updates > self.max_iters
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
    
    
class OneVsAll():
    
    def __init__(self, n_classes=9, num_features = 2048, learning_rate = 0.001, max_iters=10):
        
        self.num_classes = n_classes
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.w = np.zeros((self.num_classes, self.num_features+1), dtype=np.float64) ### one weight for each class first term is bias
        self.train_loss = []
        self.val_loss = []
        self.max_iters = max_iters
        self.num_updates = 0
        
    
    def add_bias_column(self, X):
        ### add the bias column of all ones to the X (as first column)
        ### transform it form N*d => N*d+1 
        ones = np.ones((len(X), 1), dtype=np.float64) #### N * 1
        one_X = np.concatenate((ones, X), axis=1)
        return one_X    

    def training_finished(self):
        
        return self.num_updates > self.max_iters
    
    def class_predict(self, X, w_class):
        ### X is N * d+1
        logits = np.dot(X, w_class)
        prob = np.exp(logits) / (1 + np.exp(logits))
        return prob
        
    def pred(self, X):
        ### predicts the class
        X = self.add_bias_column(X)
        logits = np.matmul(X, np.transpose(self.w)) ###  N * c
        class_pred = np.argmax(logits, axis = 1) + 1
        return class_pred
    
    def loss(self, X, y):
        
        ### finds average loss across all classifiers
        
        total_loss = 0.0
        
        for class_idx in range(1, self.num_classes+1):
            w_class = self.w[class_idx-1]
            class_pred = self.class_predict(X, w_class)
            y_class_binary = (y==class_idx).astype(np.float64)
            class_loss =  -np.mean( y_class_binary * np.log(class_pred) + (1-y_class_binary)*np.log(1-class_pred))
            total_loss += class_loss
            
        return total_loss / self.num_classes
    
    def custom_loss(self, X, y, loss=None):
        
        ### for code compatibility purposes, loss parameter value doesn't matter
        X = self.add_bias_column(X)
        return self.loss(X, y)
        

    def class_gradient(self, X, y_class, w_class):
        
        ### same form as linear regression 
        pred = self.class_predict(X, w_class)
        error = pred - y_class
        error = np.reshape(error, newshape=(-1, 1))
        grad_matrix = X * error
        grad = np.mean(grad_matrix, axis=0)
        return grad
    
    
    def update_weights(self, train_X, train_y, val_X, val_y):
        
        
        train_X = self.add_bias_column(train_X)
        val_X = self.add_bias_column(val_X)
        
        if(self.num_updates == 0):
            self.train_loss.append(self.loss(train_X, train_y))
            self.val_loss.append(self.loss(val_X, val_y))
        
        ### gradient update loop for each classifier
        for class_idx in range(1, self.num_classes+1):
            
            train_y_binary = (train_y == class_idx).astype(np.float64)
            w_class = self.w[class_idx-1]
            class_gradient = self.class_gradient(train_X, train_y_binary, w_class)
            self.w[class_idx-1] = self.w[class_idx-1] - self.learning_rate * class_gradient
            
        self.num_updates += 1
        
        self.train_loss.append(self.loss(train_X, train_y))
        self.val_loss.append(self.loss(val_X, val_y))          


class LinearClassifier():
    
    def __init__(self, n_classes, num_features, learning_rate=0.001, max_iters=10):
        
        
        self.num_classes = n_classes
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.num_updates = 0
        self.train_loss = []
        self.val_loss = []
        self.w = np.zeros((self.num_classes-1, self.num_features+1), dtype=np.float64) ### for C classes we require only C-1 weights 
        
    def add_bias_column(self, X):
        ### add the bias column of all ones to the X (as first column)
        ### transform it form N*d => N*d+1 
        ones = np.ones((len(X), 1), dtype=np.float64) #### N * 1
        one_X = np.concatenate((ones, X), axis=1)
        return one_X    

    def training_finished(self):
        
        return self.num_updates > self.max_iters
    
    
    def class_prob_predict(self, X, class_idx):
        
        denominator = np.ones(shape=(len(X),), dtype=np.float64) ## N, 
        
        for i in range(1, self.num_classes): ### from 1 to C-1
            denominator += np.exp(np.dot(X, self.w[i-1]))
        
        if ( class_idx == self.num_classes ) :
            ### 1- prediction of all classes
            numerator = np.ones(shape=(len(X,)), dtype=np.float64)
        else:
            numerator = np.exp(np.dot(X, self.w[class_idx-1]))
            
        return numerator / denominator
    
    def class_prob_predict_single(self, x, class_idx):
        ## x is a single vector
        denominator = 1.0  
        
        for i in range(1, self.num_classes): ### from 1 to C-1
            denominator += np.exp(np.dot(x, self.w[i-1]))
        
        if ( class_idx == self.num_classes ) :
            ### 1- prediction of all classes
            numerator = 1.0
        else:
            numerator = np.exp(np.dot(x, self.w[class_idx-1]))
            
        return numerator / denominator
    
    def loss(self, X, y):
        ### X is N*d+1 and y is N
        loss = 0.0
        for i in range(len(X)):
            loss += -np.log(self.class_prob_predict_single(X[i],y[i]))
        return loss / len(X)
        
    def custom_loss(self, X, y, loss=None):
        #### for code compatibility loss is redudant
        return self.loss(self.add_bias_column(X), y)
    
    def pred(self, X):
        
        X = self.add_bias_column(X)
        logits = np.matmul(X, np.tranpose(self.w))
        logits = np.concat((logits, np.zeros((len(X),1))), axis=1)
        return np.argmax(logits, axis=1) + 1
    
    def gradient_update_per_class(self, X, y, class_idx):
        #####
        prob_pred = self.class_prob_predict(X, class_idx)
        binarized_y = (y==class_idx)
        error = prob_pred - binarized_y
        error = np.reshape(error, newshape=(-1,1))
        grad_matrix = X*error
        grad = np.mean(grad_matrix, axis=0)
        return grad
    
    def update_weights(self, train_X, train_y, val_X, val_y):
        
        train_X = self.add_bias_column(train_X)
        val_X = self.add_bias_column(val_X)
        
        if(self.num_updates==0):
            self.train_loss.append(self.loss(train_X, train_y))
            self.val_loss.append(self.loss(val_X, val_y))
            
            
        ### gradient update loop
        ### updates each gradient in the w matrix
        for i in range(len(self.w)):
            gradient_t = self.gradient_update_per_class(train_X, train_y, i+1)
            self.w[i] = self.w[i] - self.learning_rate * gradient_t
        ########################
        
        self.num_updates += 1
        
        self.train_loss.append(self.loss(train_X, train_y))
        self.val_loss.append(self.loss(val_X, val_y))