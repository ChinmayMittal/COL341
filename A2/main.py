from svm_binary import Trainer
from svm_multiclass import Trainer_OVA, Trainer_OVO

C = 1
kernel = "linear"
num_classes = 10

classification_model = "mutli" ### multi / binary
multi_type = "oVa" ### oVo vs oVa 

if classification_model == "binary":
    train_data_path = "./bi_train.csv"
    test_data_path = "./bi_val.csv"
    ### binary classifier
    model = Trainer(C=C, kernel=kernel)
else:
    train_data_path="./multi_train.csv"
    test_data_path="./multi_val.csv"
    ### multiclass classifier
    if multi_type == "oVo":
        model = Trainer_OVO(C=C, kernel=kernel, n_classes=num_classes)
    else:
        model = Trainer_OVA(C=C, kernel=kernel, n_classes=num_classes)





model.fit(train_data_path=train_data_path)
model.predict(test_data_path=train_data_path)
model.predict(test_data_path=test_data_path)

