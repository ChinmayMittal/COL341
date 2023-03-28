from svm_binary import Trainer
from svm_multiclass import Trainer_OVA, Trainer_OVO
import matplotlib.pyplot as plt
import numpy as np
# C = .1
kwargs = {
    "poly_c" : 1,
    "poly_d" : 2,
    "rbf_gamma" : 0.01,
    "sigmoid_gamma" : 1,
    "sigmoid_r" : 1,
    "laplacian_gamma" : 0.001
}
kernel = "linear"

num_classes = 10


classification_model = "binary" ### multi / binary
multi_type = "oVa" ### oVo vs oVa 

val_acc_list = []

# gamma_list = [0.1,0.01,0.001]
gamma_list = [None]
# C_list = [0.0001, 0.001, 0.005, 0.01,0.1,1.0,10.0]
C_list = [0.1]
for gamma in gamma_list:
    val_acc_list = []
    for C in C_list:
        kwargs["rbf_gamma"] = gamma
        if classification_model == "binary":
            train_data_path = "./bi_train.csv"
            test_data_path = "./bi_val.csv"
            ### binary classifier
            model = Trainer(C=C, kernel=kernel, **kwargs)
        else:
            train_data_path="./multi_train.csv"
            test_data_path="./multi_val.csv"
            ### multiclass classifier
            if multi_type == "oVo":
                model = Trainer_OVO(C=C, kernel=kernel, n_classes=num_classes,**kwargs)
            else:
                model = Trainer_OVA(C=C, kernel=kernel, n_classes=num_classes,**kwargs)
        model.fit(train_data_path=train_data_path)
        train_acc = model.get_accuracy(test_data_path=train_data_path)
        val_acc = model.get_accuracy(test_data_path=test_data_path)
        val_acc_list.append(val_acc)
        print(f"Train Accuracy: {train_acc*100}%, Val Accuracy: {val_acc*100}%")

    plt.scatter(np.array(C_list).astype("str"), val_acc_list, marker="X", c="red")
    plt.grid(True)
    plt.xlabel("C")
    plt.ylabel("Validation Accuracy")
    if gamma is not None:
        plt.title(f"gamma: {gamma}")
    plt.show()