import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from dtree import DTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from utilities import read_data, class_name_to_label, get_metrics, read_test_data

parser = argparse.ArgumentParser(
                    prog = 'Decision Trees and Random Forests',
                    description = 'COL341 Assignment 3',
                    epilog = 'by Chinmay Mittal')

parser.add_argument("--train_path", action="store", type=str, required=True, dest="train_path", help="path to train folder")
parser.add_argument("--val_path", action="store", type=str, required=False, dest="val_path", default=None, help="path to val folder")
parser.add_argument("--test_path", action="store", type=str, required=True, dest="test_path", help="path to test folder")
parser.add_argument("--out_path", action="store", type=str, required=True, dest="out_path", help="path to store the csv file")

args = parser.parse_args()

## create output directory
if(args.out_path[-1] != "/"):
    args.out_path += "/"
output_dir =  os.path.split(args.out_path)[0]
print(f"Output Directory: {output_dir}")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

X_train, y_train = read_data(args.train_path)
if args.val_path is not None:
    X_val, y_val = read_data(args.val_path)
else:
    X_val, y_val = X_train[0:-1:5, :], y_train[0:-1:5]
X_test, test_filenames = read_test_data(args.test_path)

#### set to face classification
y_train = (y_train == class_name_to_label["person"]).astype(np.int32)
y_val = (y_val == class_name_to_label["person"]).astype(np.int32)
####

### store for later
X_train_, y_train_ = X_train, y_train
X_val_, y_val_ = X_val, y_val
X_test_ = X_test

print(f"Training Data: {X_train.shape}")
print(f"Valdiation Data: {X_val.shape}")
print(f"Test Data: {X_test.shape}")

#### params
trained = False
verbose = 2
tree_params = {
    "criterion" : "entropy",
    "max_depth" : 10,
    "min_samples_split" : 7,
}
####

for section in ["A", "B", "C", "D", "E", "F", "H"]:
    args.section = section
    trained = False
    X_train, y_train = X_train_, y_train_
    X_val, y_val = X_val_, y_val_
    X_test = X_test_
    if args.section == "A":
        print("Decision Tree implementation from scratch")
        tree_params = {
            "criterion" : "entropy",
            "max_depth" : 10,
            "min_samples_split" : 7,
        }
        model = DTree(**tree_params)

    elif args.section == "B":
        ### sk-learn decision trees
        print("Sklearn Decision Trees")
        model = DecisionTreeClassifier(**tree_params, random_state=31)
        
    elif args.section == "C":

        ### feature selection
        print("Feature Selection ... ")
        clf = DecisionTreeClassifier(random_state=0)
        feature_selector = SelectFromModel(clf, max_features=10)
        feature_selector.fit(X_train, y_train)
        selected_features = feature_selector.get_support()
        X_train = X_train[:, selected_features]
        X_val = X_val[:, selected_features]
        X_test = X_test[:, selected_features]
        ###[HARD CODED] parameters learnt from grid search
        model = DecisionTreeClassifier(random_state=0, criterion="entropy", max_depth=None, min_samples_split=4)
        print(f"Number of Features Selected: {np.sum(selected_features)}")

    elif args.section == "D":
        print("Cost Complexity Pruning ... ")
        ### [HARD CODED] optimal alpha is hard code from validation
        model = DecisionTreeClassifier(random_state=0, ccp_alpha=0.00274)
        # path = model.cost_complexity_pruning_path(X_train, y_train)
        # ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # clfs = []
        # for i, ccp_alpha in enumerate(ccp_alphas):
        #     print(f"Training Pruned Trees {(i/len(ccp_alphas))*100:.2f}%", end="\r")
        #     clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        #     clf.fit(X_train, y_train)
        #     clfs.append(clf)
        # clfs, ccp_alphas = clfs[:-1], ccp_alphas[:-1] ### last tree is trivial tree with single node
        # node_counts = [clf.tree_.node_count for clf in clfs]
        # depths = [clf.tree_.max_depth for clf in clfs]
        # train_scores = [clf.score(X_train, y_train) for clf in clfs]
        # val_scores = [clf.score(X_val, y_val) for clf in clfs]
        # max_val_index = np.array(val_scores).argmax()
        # model = clfs[max_val_index]
        # trained = True
        
    elif args.section == "E":
        ### random forests
        print("Random Forests ... ")
        ###[HARD CODED] parameters learnt from grid search
        model = RandomForestClassifier(random_state=0, n_estimators=100, criterion="entropy", min_samples_split=5, max_depth=None)
        
    elif args.section == "F":
        print("Gradient Boosted Trees and XGBoost ... ")
        # model = GradientBoostingClassifier(random_state=0)
        ###[HARD CODED] parameters learnt from grid search || ONLY XGBoost for this part
        model = XGBClassifier(random_state=0, n_estimators=50, max_depth=5, subsample=0.5)
        
    elif args.section == "H":
        print("Competitive Part ... ")
        ##[TODO] best model from all experiments
        model = XGBClassifier(random_state=0, n_estimators=50, max_depth=5, subsample=0.5)

    ### TRAINING
    if not trained:
        training_start_time = time.perf_counter()
        model.fit(X_train, y_train)
        training_end_time = time.perf_counter()
        training_time_taken_ms = (training_end_time - training_start_time) * 1000
        print(f"Training Time Section: {args.section} Time: {training_time_taken_ms:.4f} ms")

    ### INFERENCE
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc, train_precision, train_recall = get_metrics(y_true=y_train, y_pred=train_pred)
    val_acc, val_precision, val_recall = get_metrics(y_true=y_val, y_pred=val_pred)

    ### METRICS
    print()
    print(f"Training Metrics {args.section}  | Accuracy: {train_acc:.4f}% | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
    print(f"Validation Metrics {args.section}| Accuracy: {val_acc:.4f}% | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

    ### TESTING
    test_pred = model.predict(X_test)
    test_labels = [f"{test_filename.split('.')[0]}," for test_filename in test_filenames]
    df = pd.DataFrame(data={
        "labels" : test_labels,
        "pred" : test_pred.tolist()
    })
    if ".csv" not in args.out_path:
        output_path = os.path.join(args.out_path, f"test_31{args.section.lower()}.csv")
    df.to_csv(output_path, header=False, index=False, sep=" ")



