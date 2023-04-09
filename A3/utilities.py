import os
import cv2
import typing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix

class_name_to_label = {
    "car" : 0,
    "person" : 1,
    "airplane" : 2,
    "dog" : 3,
}

def read_data(path:str):
    """_summary_

    Args:
        path (str): path to the folder
        
    """
    image_list = []
    label_list = []
    sub_dirs = [os.path.join(path, subdir) for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]
    for sub_dir in sub_dirs:
        class_name = sub_dir.split("/")[-1]
        for image in os.listdir(sub_dir):
            if image.endswith('.jpg') or image.endswith('.png'):
                image_path = os.path.join(sub_dir, image)
                img= cv2.imread(image_path)
                img_array = np.array(img)
                image_list.append(np.reshape(img_array, newshape=(1,-1)))
                label_list.append(class_name_to_label[class_name])
    
    X = np.vstack(image_list)
    y = np.array(label_list, dtype=np.int32)
    return X, y
            
def read_test_data(test_path:str):
    image_list = []
    test_filenames = os.listdir(test_path)
    test_filenames.sort(key = lambda filename : int(filename.split(".")[0].split("_")[-1]) )
    for image in test_filenames:
        if image.endswith('.jpg') or image.endswith('.png'):
            image_path = os.path.join(test_path, image)
            img = cv2.imread(image_path)
            img_array = np.array(img)
            image_list.append(np.reshape(img_array, newshape=(1,-1)))
    
    return np.vstack(image_list), test_filenames
        
        
def get_accuracy(y_true, y_pred):
    return (np.sum(y_pred == y_true)/y_true.shape[0]) * 100

def get_metrics(y_true, y_pred, average="binary"):
    ### accuracy, precision, recall
    return get_accuracy(y_true, y_pred), precision_score(y_true, y_pred, average=average), recall_score(y_true, y_pred, average=average)        
    
def plot_confusion_matrix(y_true, y_pred, labels):
    cf = confusion_matrix(y_true.astype(np.int32), y_pred.astype(np.int32))
    confusion_df = pd.DataFrame(cf, index = labels, columns= labels)
    plt.figure(figsize=(7,7))
    sns.heatmap(confusion_df, annot=True, fmt=".0f", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Prediction Label")
    plt.show()