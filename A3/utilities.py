import typing
import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score

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
            
def get_accuracy(y_true, y_pred):
    return (np.sum(y_pred == y_true)/y_true.shape[0]) * 100

def get_metrics(y_true, y_pred):
    ### accuracy, precision, recall
    return get_accuracy(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)        
    