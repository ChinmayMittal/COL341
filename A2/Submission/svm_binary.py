from typing import List
import numpy as np
# import qpsolvers
from qpsolvers import solve_qp
from utilities import read_file, plot_confusion_helper
from kernel import get_k_mat, get_k_mat_

eps = 1e-4

def get_pred_signal(x, support_alphas, support_ys, support_xs, b, kernel, **kwargs):
    
    kernel_values = get_k_mat_(x, support_xs, kernel, **kwargs)
    return b + np.sum(support_ys * support_alphas * kernel_values[0])

class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        
    
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        X,y = read_file(train_data_path) ### utility to read the dataset
        self.fit_helper(X,y) ### helper function to find the support vectors
        
        
    def fit_helper(self, X, y):
        
        ### convert y to +1 / -1
        self.X_train, self.y_train = X, y
        self.y_train = (self.y_train*2-1)
        
        self.N_train = self.X_train.shape[0]
        K_mat = get_k_mat(self.X_train, self.kernel, **self.kwargs)

        y = np.reshape(self.y_train, newshape=(self.N_train,1))
        y_mat = np.hstack([y]*self.N_train)
        y_mat = y_mat * (y_mat.T) #### y_mat[i][j] = y_i * y_j
        P = (y_mat * K_mat)
        q = -1*np.ones(shape=(self.N_train,))
        G, h  = None, None
        lb = np.zeros(shape=(self.N_train,))  ## alpha > 0 
        ub = self.C * np.ones(shape=(self.N_train,)) ### alpha < C
        A = np.array(self.y_train) ### sum alpha_i * y_i = 0
        b = np.array([0.0])
        
        alpha = solve_qp(P=P + 1e-7*np.identity(P.shape[0]), q=q, G=G, h=h, A=A, b=b, lb=lb, ub=ub, solver="ecos")
        b_mask = np.logical_and( alpha > eps, alpha < self.C - eps ) ###  0 < alpha < C
        
        support_vector_mask = alpha > eps ### all the support vectors
        
        self.y_support = self.y_train[support_vector_mask] ### will be used for inference
        self.x_support = self.X_train[support_vector_mask, :] 
        for support_vec in self.x_support:
            self.support_vectors.append(support_vec)
        self.alpha_support = alpha[support_vector_mask] ### will be used for inference
        
        ### use this data point to compute bias ## (any point with 0 < alpha < C)
        b_ind = (np.argmax(b_mask)) 
        b_vector = self.X_train[[b_ind], :]
        ######################################

        #### utility to get the prediction signal which takes dot product in the kernel space with the support vectors
        signal = get_pred_signal(x=b_vector, support_alphas=self.alpha_support, support_xs=self.x_support, support_ys=self.y_support, b = 0 , kernel=self.kernel, **self.kwargs)
        self.b = self.y_train[b_ind] - signal ### store the bias for inference
        
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        X, y = read_file(test_data_path)
        return self.predict_helper(X,y,raw_signal=False)
        
    def predict_helper(self,X,y,raw_signal=False):

        n = X.shape[0]
        y_pred = []
        for i in range(n):
            signal = get_pred_signal(x=X[[i],:], support_alphas=self.alpha_support, support_ys=self.y_support, support_xs=self.x_support, b=self.b, kernel=self.kernel, **self.kwargs)
            if not raw_signal:
                y_pred.append( 1 if signal > 0 else 0 )
            else:
                y_pred.append(signal)
        return np.asarray(y_pred, dtype=np.float32)
    
    def get_accuracy(self,test_data_path:str, plot_confusion=False):
        ### helper function to compute accuracy
        X, y = read_file(test_data_path)
        y_pred = self.predict_helper(X,y,raw_signal=False)
        if plot_confusion: ### utility to plot confusion matrix 
            plot_confusion_helper(y,y_pred) 
        return np.sum(np.array(y_pred) == y) / X.shape[0]
    
    