import numpy as np

class AdamOptimizer():
    
    def __init__(self, shape, alpha, beta1=0.9, beta2=0.99, eps=1e-8): 
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.num_updates = 0
        self.v_dw = np.zeros(shape)
        self.s_dw = np.zeros(shape)
        
    def update(self, w, w_grad, printing=False):
        
        self.num_updates += 1
        
        self.v_dw = self.beta1 * self.v_dw + (1-self.beta1)*w_grad
        self.s_dw = self.beta2 * self.s_dw + (1-self.beta2)*(w_grad ** 2)
        
        v_dw_correct = self.v_dw / (1-(self.beta1**self.num_updates))
        s_dw_correct = self.s_dw / (1-(self.beta2**self.num_updates))
        if(printing):
            print(self.alpha * (v_dw_correct / (np.sqrt(s_dw_correct) + self.eps)))
        return w - self.alpha * (v_dw_correct / (np.sqrt(s_dw_correct) + self.eps) )