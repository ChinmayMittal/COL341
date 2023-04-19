from layers import Linear, ReLU, SoftMax

class CNN():
    
    def __init__(self, NUM_CLASSES=10):
        self.FC1 = Linear(in_features=3072, out_features=64)
        self.ReLU1 = ReLU()
        self.FC2 = Linear(in_features=64, out_features=NUM_CLASSES)
        self.softmax = SoftMax()
    
    def forward(self, X):
        ### X is M * 3072
        X = self.FC1.forward(X)
        X = self.ReLU1.forward(X)
        X = self.FC2.forward(X)
        X = self.softmax.forward(X)
        return X
    
    def backward(self, dep_prob):
        ### del_prob => M * NUM_CLASSES => each term is gradient wrt loss
        delJ = self.softmax.backward(dep_prob)
        delJ = self.FC2.backward(delJ)
        delJ = self.ReLU1.backward(delJ)
        delJ = self.FC1.backward(delJ)
    
    def update_weights(self, lr):
        self.FC2.update(lr=lr)
        self.FC1.update(lr=lr)
