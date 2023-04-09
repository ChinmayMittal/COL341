import numpy as np

def majority_class(y):
    ### returns the majority class in the numpy array y
    if(len(y) ==  0):
        return 0 ### default class in case node has no training samples
    values, counts = np.unique(y, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def gini_helper(x):
    total_size = x.shape[0]
    if(total_size==0):
        return 0
    zero_count, one_count = np.sum(x==0), np.sum(x==1)
    return (1 - ((zero_count/total_size)**2 + (one_count/total_size)**2) )

def get_gini_gain(x1, x2):
    total_size = x1.shape[0] + x2.shape[0]
    return  ((x1.shape[0]/total_size)* gini_helper(x1)) +  (( x2.shape[0] / total_size)* gini_helper(x2))
    

def best_threshold(x, y):
    ### x is a feature, y is the output label
    ### returns the gain and the best threshold
    f_values = np.unique(x).tolist()
    f_values = [f_values[0]-1] + f_values
    best_gain, best_threshold = None, None
    for split_value in f_values:
        split_mask = (x <= split_value)
        gain = get_gini_gain(y[split_mask], y[np.logical_not(split_mask)])
        if((best_gain is None) or (gain < best_gain)):
            best_gain, best_threshold = gain, split_value
    
    return best_gain, best_threshold

### DTREE NODE
class Node:
    
    def __init__(self, isLeaf=True, prediction=0, feature_index=0, feature_value=0):
        self.isLeaf = isLeaf
        self.prediction = prediction ### prediction if this node is a leaf
        self.leftNode, self.rightNode = None, None
        self.feature_index = feature_index ### use this feature index to decide
        self.feature_value = feature_value ### if feature <= feature_value => go left, else right

    def predict(self, X):
        ### X is NUM_SAMPLES * NUM_FEATURES
        if self.isLeaf: ### all inputs will have same prediction
            return self.prediction * np.ones(shape=(X.shape[0],))
        prediction = -1 * np.ones(shape=(X.shape[0])) ### dummy initialization
        left_mask = X[:,self.feature_index] <= self.feature_value
        right_mask = np.logical_not(left_mask)
        ### left child
        if(np.any(left_mask)):
            prediction[left_mask] = self.leftNode.predict(X[left_mask, :])
        if(np.any(right_mask)):
            prediction[right_mask] = self.rightNode.predict(X[right_mask, :])
        return prediction

### implementing Decision Trees from Scratch
class DTree:
    
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def check_base_case(self, X, y, cur_depth):
                        #### max depth is set and we have reached max depth             too few samples to split
        return ( (self.max_depth is not None) and (cur_depth == self.max_depth) ) or (X.shape[0] < self.min_samples_split)
    
    
    def find_best_feature(self, X, y):
        ### returns feature index, feature value
        best_feature_gain,  best_feature_idx, best_feature_value  = None, None, None
        
        for f_idx in range(X.shape[1]):
            ### iterate over all features
            feature_gain, feature_threshold = best_threshold(X[:,f_idx], y)
            if (best_feature_gain is None) or (feature_gain < best_feature_gain):
                best_feature_gain, best_feature_idx, best_feature_value = feature_gain, f_idx, feature_threshold
                
        return best_feature_idx, best_feature_value
            
    def trainer(self, X, y, cur_depth):
        #### returns the root (class Node) when trained on X, y
        
        ### base case
        if(self.check_base_case(X,y,cur_depth)):
            return Node(isLeaf=True, prediction=majority_class(y))
        
        ### find best feature
        best_feature_index, best_feature_value = self.find_best_feature(X,y) ### the index for the feature to consider and the corresponding feature value to create the split
        left_mask = X[:, best_feature_index] <= best_feature_value
        right_mask = np.logical_not(left_mask)
        
        ### recurse
        rootNode = Node(isLeaf=False, prediction=-1, feature_index=best_feature_index, feature_value=best_feature_value)
        rootNode.leftNode = self.trainer(X[left_mask, :], y[left_mask], cur_depth+1)
        rootNode.rightNode = self.trainer(X[right_mask, :], y[right_mask], cur_depth+1)
        
        return rootNode
        
    def fit(self, X, y):
        self.root = self.trainer(X,y, cur_depth=0)
    
    def predict(self, X):
        return self.root.predict(X)
    

    
