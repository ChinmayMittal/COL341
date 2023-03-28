import pandas as pd
import operator as op
import numpy as np
from functools import reduce

def read_file(path):
    df = pd.read_csv(path)
    y = df.iloc[:, -1].values
    X = df.iloc[:, 1:-1].values
    return X, y

def read_file_multi(path):
    df = pd.read_csv(path)
    y = df.iloc[:,1].values.astype(np.int32)
    X = df.iloc[:,2:].values
    return X,y


def nCr(n, r):
    r = min(r, n-r)
    num = reduce(op.mul, range(n, n-r, -1), 1)
    den = reduce(op.mul, range(1, r+1), 1)
    return num // den


