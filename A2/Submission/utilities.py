import pandas as pd
import operator as op
import numpy as np
from functools import reduce
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def read_file(path):
    df = pd.read_csv(path)
    columns = df.columns.tolist()
    if("y" in columns or "Y" in columns):
        y = df.iloc[:, -1].values
        X = df.iloc[:, 1:-1].values
    else:
        X = df.iloc[:, 1:].values
        y = None
    return X, y

def read_file_multi(path):
    df = pd.read_csv(path)
    columns = df.columns.tolist()
    if("y" in columns or "Y" in columns):
        y = df.iloc[:,1].values.astype(np.int32)
        X = df.iloc[:,2:].values
    else:
        y = None
        X = df.iloc[:,1:].values
    return X,y

def plot_confusion_helper(y_true, y_pred):
    import seaborn
    cf_matrix = confusion_matrix(y_true, y_pred)
    if cf_matrix.shape[0] == 2:
        labels = [i for i in "01"]
    else:
        labels = [str(i) for i in range(1,cf_matrix.shape[0]+1)]
    df_cm = pd.DataFrame(cf_matrix, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True,cmap ='viridis')
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

