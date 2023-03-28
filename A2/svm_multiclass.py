from typing import List
import numpy as np
from svm_binary import Trainer
from utilities import read_file_multi

class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        # self.svms[i][j] is the svm for the pair i,j if i < j
        for i in range(self.n_classes):
            svm_per_class_list = []
            for j in range(self.n_classes):
                svm_per_class_list.append(Trainer(C=self.C, kernel=self.kernel, **self.kwargs) if j > i else None)  
            self.svms.append(svm_per_class_list)
    
            
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        X,y = read_file_multi(train_data_path)
        self._init_trainers()
        
        for i in range(self.n_classes):
            for j in range(i+1, self.n_classes):
                i_mask, j_mask = (y==i+1), (y==j+1)
                X_i, X_j = X[i_mask, :], X[j_mask, :]
                X_combined = np.vstack((X_i, X_j))
                y_i, y_j = np.ones(shape=(X_i.shape[0])), np.zeros(shape=(X_j.shape[0]))
                y_combined = np.concatenate([y_i,y_j])
                self.svms[i][j].fit_helper(X_combined, y_combined)
        
        
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        X,y = read_file_multi(test_data_path)
        self.X_test, self.y_test = X,y
        n = X.shape[0]
        votes = np.zeros(shape=(X.shape[0], self.n_classes))
        for i in range(self.n_classes):
            for j in range(i+1, self.n_classes):
                pred = self.svms[i][j].predict_helper(X,y=None, raw_signal=False)### one if i+1 else zero if j+1
                # print(pred)
                votes[:,i] += pred
                votes[:,j] += (1-pred)
        y_pred = (np.argmax(votes, axis=1)+1)
        # print(np.sum(np.array(y_pred) == y) / n)
        return y_pred        
        
    def get_accuracy(self,test_data_path:str):
        y_pred = self.predict(test_data_path)
        return np.sum(np.array(y_pred) == self.y_test) / self.X_test.shape[0]       
    
class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        ### one trainer for each class
        for i in range(self.n_classes):
            self.svms.append(Trainer(C=self.C, kernel=self.kernel, **self.kwargs))
            
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        X,y = read_file_multi(train_data_path)
        self._init_trainers()
        for i in range(self.n_classes):
            ## fit a trainer for each class
            self.svms[i].fit_helper(X, (y==i+1).astype(np.int32))
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
       X,y = read_file_multi(test_data_path)
       self.X_test, self.y_test = X,y
       n = X.shape[0]
       pred_list = []
       for i in range(self.n_classes):
           pred = self.svms[i].predict_helper(X=X, y=None, raw_signal=True)
           pred_list.append(np.reshape(pred, newshape=(X.shape[0],1)))
       signal_pred = np.hstack(pred_list)
       y_pred = (np.argmax(signal_pred, axis=1)+1)
    #    print(np.sum(np.array(y_pred) == y) / n)
       return y_pred
       
    def get_accuracy(self,test_data_path:str):
        y_pred = self.predict(test_data_path)
        return np.sum(np.array(y_pred) == self.y_test) / self.X_test.shape[0]