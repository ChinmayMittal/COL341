import numpy as np
from layers import MaxPool2D, Conv2D


layer = Conv2D(kernel_size=3, stride=1, in_channels=3, num_filters=32)
input = np.random.randint(low=0, high=10, size = (16,3,32,32))
output = layer.forward(input)
print(output.shape)
