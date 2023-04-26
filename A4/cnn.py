from layers import Linear, ReLU, SoftMax, Flatten, MaxPool2D, Conv2D

class CNN():
    
    def __init__(self, NUM_CLASSES=10, learning_rate=1e-2):
        
        self.NUM_CLASSES = NUM_CLASSES
        self.learning_rate = learning_rate
        
        self.conv1 = Conv2D(kernel_size=3, stride=1, in_channels=3, num_filters=32, learning_rate=learning_rate)
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = Conv2D(kernel_size=5, stride=1, in_channels=32, num_filters=64, learning_rate=learning_rate)
        self.ReLU2 = ReLU()
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        
        self.conv3 = Conv2D(kernel_size=3, stride=1, in_channels=64, num_filters=64, learning_rate=learning_rate)
        self.ReLU3 = ReLU()
        
        self.flatten = Flatten()
        
        self.FC1 = Linear(in_features=3*3*64, out_features=64, learning_rate=learning_rate)
        self.ReLU4 = ReLU()
        
        self.FC2 = Linear(in_features=64, out_features=NUM_CLASSES, learning_rate=learning_rate)
        self.softmax = SoftMax()
    
    def forward(self, X):
        ### X is M * C * H * W
        X = self.conv1.forward(X)
        X = self.ReLU1.forward(X)
        X = self.pool1.forward(X)
        
        X = self.conv2.forward(X)
        X = self.ReLU2.forward(X)
        X = self.pool2.forward(X)
        
        X = self.conv3.forward(X)
        X = self.ReLU3.forward(X)
        
        X = self.flatten.forward(X)
        
        X = self.FC1.forward(X)
        X = self.ReLU4.forward(X)
        
        X = self.FC2.forward(X)
        X = self.softmax.forward(X)
        
        return X
    
    def backward(self, dep_prob):
        ### del_prob => M * NUM_CLASSES => each term is gradient wrt loss
        delJ = self.softmax.backward(dep_prob)
        delJ = self.FC2.backward(delJ)
        
        delJ = self.ReLU4.backward(delJ)
        delJ = self.FC1.backward(delJ)
        
        delJ = self.flatten.backward(delJ)
        
        delJ = self.ReLU3.backward(delJ)
        delJ = self.conv3.backward(delJ)
        
        delJ = self.pool2.backward(delJ)
        delJ = self.ReLU2.backward(delJ)
        delJ = self.conv2.backward(delJ)
        
        delJ = self.pool1.backward(delJ)
        delJ = self.ReLU1.backward(delJ)
        delJ = self.conv1.backward(delJ)
        
    def update_weights(self):
        
        self.FC2.update()
        self.FC1.update()
        self.conv3.update()
        self.conv2.update()
        self.conv1.update()