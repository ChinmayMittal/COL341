import os
import pickle
import itertools
import numpy as np
from math import ceil
from constants import classes
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict    

class Dataset:
    def __init__(self, filepath, batch_size=32):
        self.filepath = filepath
        self.batch_size = batch_size
        all_files = os.listdir(self.filepath)
        train_files = [os.path.join(self.filepath, file) for file in all_files if file[:4] == "data"]
        test_files = [os.path.join(self.filepath, file) for file in all_files if file[:4] == "test"]
        train_dict = [unpickle(train_file) for train_file in train_files]
        test_dict = [unpickle(test_file) for test_file in test_files]
        train_X_list = [data_dict[b"data"] for data_dict in train_dict]
        train_y_list = [data_dict[b"labels"] for data_dict in train_dict]
        test_X_list = [data_dict[b"data"] for data_dict in test_dict]
        test_y_list = [data_dict[b"labels"] for data_dict in test_dict]
        self.X_train = np.vstack(train_X_list)
        self.X_test = np.vstack(test_X_list)
        self.y_train = np.array(list(itertools.chain(*train_y_list)))
        self.y_test = np.array(list(itertools.chain(*test_y_list)))
        self.num_train_samples, self.num_test_samples = self.X_train.shape[0], self.X_test.shape[0]
        self.num_train_batches =  ceil(self.num_train_samples/self.batch_size)
        self.num_test_batches = ceil(self.num_test_samples/self.batch_size )
        self.train_mean, self.train_std = np.mean(self.X_train, axis=0), np.std(self.X_train, axis=0)

    def visualize(self, idx):
        plt.imshow(np.moveaxis(self.X_train[idx].reshape((3,32,32)), source=0, destination=-1))
        plt.title(classes[self.y_train[idx]])
        plt.show()

    def train_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx+1)*self.batch_size, self.num_train_samples)
        X, y = self.X_train[start_idx:end_idx, :], self.y_train[start_idx:end_idx]
        X = (X-self.train_mean)/self.train_std
        return X, y
    
    def test_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx+1)*self.batch_size, self.num_test_samples)
        X, y = self.X_test[start_idx:end_idx, :], self.y_test[start_idx:end_idx]
        X = (X - self.train_mean)/self.train_std
        return X, y
        
# 

