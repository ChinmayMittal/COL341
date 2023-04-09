from typing import Union
from svm_binary import Trainer as T
from svm_multiclass import Trainer_OVO as T_ovo, Trainer_OVA as T_ova

def best_classifier_two_class()->T:
    """Return the best classifier for the two-class classification problem."""
    #TODO: implement, use best performing values for C, kernel functions and all the parameters of the kernel functions
    # Set Hyper-params
    kwargs = {
        "poly_c" : 1,
        "poly_d" : 2,
        "rbf_gamma" : 0.01,
        "sigmoid_gamma" : 1,
        "sigmoid_r" : 1,
        "laplacian_gamma" : 0.001
    }
    trainer = T(kernel="linear", C=0.001, **kwargs)
    return trainer

def best_classifier_multi_class()->Union[T_ovo,T_ova]:
    """Return the best classifier for the multi-class classification problem."""
    #TODO: implement, use best performing model with optimum values for C, kernel functions and all the parameters of the kernel functions.
    # Set Hyper-params
    # Set the trainer to either of T_ovo or T_ova
    # Create trainer with hyper-parameters
    kwargs = {
        "poly_c" : 1,
        "poly_d" : 2,
        "rbf_gamma" : 0.01,
        "sigmoid_gamma" : 1,
        "sigmoid_r" : 1,
        "laplacian_gamma" : 0.001
    }
    kwargs["rbf_gamma"] = 0.001
    trainer = T_ovo(kernel="rbf", C=25, n_classes=10,  **kwargs)
    return trainer
