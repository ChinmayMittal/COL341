from svm_binary import Trainer
from svm_multiclass import Trainer_OVA, Trainer_OVO
from best import best_classifier_multi_class, best_classifier_two_class
import matplotlib.pyplot as plt
import numpy as np

kwargs = {
    "poly_c" : 1,
    "poly_d" : 2,
    "rbf_gamma" : 0.01,
    "sigmoid_gamma" : 1,
    "sigmoid_r" : 1,
    "laplacian_gamma" : 0.001
}
kernel = "rbf"

num_classes = 10

best = False

classification_model = "multi" ### multi / binary
multi_type = "oVo" ### oVo vs oVa 

plotting = False
confusion_plot = True

val_acc_list = []

# gamma_list = [0.1]
# gamma_list = [None]
gamma_list = [0.1]
# C_list = [0.005, 0.001, 0.01,0.1,1.0,10.0, 15, 25, 35,50,75]
C_list = [0.1]
for gamma in gamma_list:
    val_acc_list = []
    for C in C_list:
        kwargs["rbf_gamma"] = gamma
        # kwargs["laplacian_gamma"] = gamma
        # kwargs["sigmoid_r"] = C
        # kwargs["sigmoid_gamma"] = gamma
        if classification_model == "binary":
            train_data_path = "./bi_train.csv"
            test_data_path = "./bi_val.csv"
            ### binary classifier
            if best:
                model = best_classifier_two_class()
            else:
                model = Trainer(C=C, kernel=kernel, **kwargs)
        else:
            train_data_path="./multi_train.csv"
            test_data_path="./multi_val.csv"
            ### multiclass classifier
            if best:
                model = best_classifier_multi_class()
            else:
                if multi_type == "oVo":
                    model = Trainer_OVO(C=C, kernel=kernel, n_classes=num_classes,**kwargs)
                else:
                    model = Trainer_OVA(C=C, kernel=kernel, n_classes=num_classes,**kwargs)
        model.fit(train_data_path=train_data_path)
        # print(model.support_vectors)
        train_acc = model.get_accuracy(test_data_path=train_data_path)
        val_acc = model.get_accuracy(test_data_path=test_data_path, plot_confusion=confusion_plot)
        val_acc_list.append(val_acc*100)
        print(f"Train Accuracy: {train_acc*100}%, Val Accuracy: {val_acc*100}%, gamma: {gamma}, C: {C}")
    
    if plotting:
        plt.scatter(np.array(C_list).astype("str"), val_acc_list, marker="X", c="red")
        plt.grid(True)
        plt.xlabel("C")
        plt.ylabel("Validation Accuracy %")
        if kernel == "rbf":
            plt.title(f"RBF kernel gamma: {gamma}, varying C")
        elif kernel == "linear":
            plt.title(f"Linear Kernel, varying C")
        plt.show()