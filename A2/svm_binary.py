from typing import List
import numpy as np
# import qpsolvers
from qpsolvers import solve_qp
from utilities import read_file
from kernel import get_k_mat

eps = 1e-5

def get_pred_signal(x, support_alphas, support_ys, support_xs, b, kernel):
    
    temp_x = np.vstack((support_xs, x))
    temp_k_mat = get_k_mat(temp_x, kernel)
    
    return b + np.sum(support_ys * support_alphas * temp_k_mat[-1,:-1])

class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        self.support_vectors:List[np.ndarray] = []
        
    
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        X,y = read_file(train_data_path)
        self.fit_helper(X,y)
        
        
    def fit_helper(self, X, y):
        
        ### convert y to +1 / -1
        self.X_train, self.y_train = X, y
        self.y_train = (self.y_train*2-1)
        
        self.N_train = self.X_train.shape[0]
        K_mat = get_k_mat(self.X_train, self.kernel)

        y = np.reshape(self.y_train, newshape=(self.N_train,1))
        y_mat = np.hstack([y]*self.N_train)
        y_mat = y_mat * (y_mat.T)
        P = (y_mat * K_mat)/2
        q = -1*np.ones(shape=(self.N_train,))
        G, h  = None, None
        lb = np.zeros(shape=(self.N_train,))
        ub = self.C * np.ones(shape=(self.N_train,))
        A = np.array(self.y_train)
        b = np.array([0.0])
        
        alpha = solve_qp(P=P, q=q, G=G, h=h, A=A, b=b, lb=lb, ub=ub, solver="osqp")
        
        b_mask = np.logical_and( alpha > eps, alpha < self.C - eps ) ###  0 < alpha < C
        
        support_vector_mask = alpha > eps
        
        self.y_support = self.y_train[support_vector_mask]
        self.x_support = self.X_train[support_vector_mask, :]
        self.alpha_support = alpha[support_vector_mask]
        
        ### use this data point to compute bias ##
        b_ind = (np.argmax(b_mask)) 
        b_vector = self.X_train[[b_ind], :]
        ######################################

        signal = get_pred_signal(x=b_vector, support_alphas=self.alpha_support, support_xs=self.x_support, support_ys=self.y_support, b = 0 , kernel=self.kernel)
        self.b = 1/self.y_train[b_ind] - signal
        
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        X, y = read_file(test_data_path)
        return self.predict_helper(X,y,raw_signal=False)
        
    def predict_helper(self,X,y,raw_signal=False):

        n = X.shape[0]
        y_pred = []
        for i in range(n):
            signal = get_pred_signal(x=X[[i],:], support_alphas=self.alpha_support, support_ys=self.y_support, support_xs=self.x_support, b=self.b, kernel=self.kernel)
            if not raw_signal:
                y_pred.append( 1 if signal > 0 else 0 )
            else:
                y_pred.append(signal)
        
        if not raw_signal and y is not None:
            print(np.sum(np.array(y_pred) == y) / n )
            
        return np.asarray(y_pred, dtype=np.float32)
    