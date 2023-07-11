import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


x = np.zeros((50,3,38,32)) ## B * C_in * H * W 
print(x.shape)
window_view = sliding_window_view(x, window_shape=(3,5,5), axis=(1,2,3))
print(window_view.shape)

