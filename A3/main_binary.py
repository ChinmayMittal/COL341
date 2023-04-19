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
from utilities import read_data, class_name_to_label, get_metrics, plot_confusion_matrix, read_test_data

parser = argparse.ArgumentParser(
                    prog = 'Decision Trees and Random Forests',
                    description = 'COL341 Assignment 3',
                    epilog = 'by Chinmay Mittal')

parser.add_argument("--train_path", action="store", type=str, required=True, dest="train_path", help="path to train folder")
parser.add_argument("--val_path", action="store", type=str, required=False, dest="val_path", default=None, help="path to val folder")
parser.add_argument("--test_path", action="store", type=str, required=True, dest="test_path", help="path to test folder")
parser.add_argument("--out_path", action="store", type=str, required=True, dest="out_path", help="path to store the csv file")
parser.add_argument("--section", action="store", type=str, required=False, default="B", dest="section", help="...")

args = parser.parse_args()

## create output directory
output_dir =  os.path.split(args.out_path)[0]
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

X_train, y_train = read_data(args.train_path)
X_val, y_val = read_data(args.val_path)
X_test, test_filenames = read_test_data(args.test_path)

#### set to face classification
y_train = (y_train == class_name_to_label["person"]).astype(np.int32)
y_val = (y_val == class_name_to_label["person"]).astype(np.int32)
####

print(f"Training Data: {X_train.shape}")
print(f"Valdiation Data: {X_val.shape}")
print(f"Test Data: {X_test.shape}")

#### params
visualization = False
grid_search = False
trained = False
plot_confusion = True
verbose = 2
tree_params = {
    "criterion" : "entropy",
    "max_depth" : 10,
    "min_samples_split" : 7,
}
####
if args.section == "A":
    print("Decision Tree implementation from scratch")
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
    model = DecisionTreeClassifier(random_state=0)
    print(f"Number of Features Selected: {np.sum(selected_features)}")
    visualization = True
    grid_search = True

elif args.section == "D":
    print("Cost Complexity Pruning ... ")
    model = DecisionTreeClassifier(random_state=0)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots(figsize=(18,8))
    vfunc = np.vectorize(lambda x : round(x,5))
    ax.plot(vfunc(ccp_alphas[:-1]).astype("str"), impurities[:-1], marker="o", drawstyle="steps-post", color="red")
    ax.set_xlabel("Effective alpha")
    ax.set_ylabel("Total impurity of leaves")
    ax.set_title("Total Impurity vs Effective alpha for Training set")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.grid(True)
    plt.show()
    plt.clf()
    clfs = []
    for i, ccp_alpha in enumerate(ccp_alphas):
        print(f"Training Pruned Trees {(i/len(ccp_alphas))*100:.2f}%", end="\r")
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    clfs, ccp_alphas = clfs[:-1], ccp_alphas[:-1] ### last tree is trivial tree with single node
    node_counts = [clf.tree_.node_count for clf in clfs]
    depths = [clf.tree_.max_depth for clf in clfs]
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    val_scores = [clf.score(X_val, y_val) for clf in clfs]
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
    ax[0].plot(vfunc(ccp_alphas).astype("str"), node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Number of nodes")
    ax[0].set_title("Number of nodes vs Alpha")
    ax[0].grid(True)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=60)
    ax[1].plot(vfunc(ccp_alphas).astype("str"), depths, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Depth of tree")
    ax[1].set_title("Depth vs Alpha")
    ax[1].grid(True)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=60)
    fig.tight_layout()
    plt.show()
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs alpha for training and validation sets")
    ax.plot(vfunc(ccp_alphas).astype("str"), train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(vfunc(ccp_alphas).astype("str"), val_scores, marker="o", label="val", drawstyle="steps-post")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.grid(True)
    ax.legend()
    plt.show()
    max_val_index = np.array(val_scores).argmax()
    model = clfs[max_val_index]
    trained = True
    
elif args.section == "E":
    ### random forests
    print("Random Forests ... ")
    model = RandomForestClassifier(random_state=0)
    grid_search = True
    
elif args.section == "F":
    print("Gradient Boosted Trees and XGBoost ... ")
    # model = GradientBoostingClassifier(random_state=0)
    model = XGBClassifier(random_state=0)
    grid_search = True
    
#### GRID SEARCH
if grid_search:
    print("Grid Search ... \n ")
    if args.section == "C":
        ### decision trees
        parameters = dict(criterion=["gini", "entropy"],
                    max_depth=[None, 5, 7, 10, 15],
                    min_samples_split=[2,4,7,9])
    elif args.section == "E":
        ### random forests
        parameters = dict( n_estimators = [80,100,150,200],
                        criterion=["gini", "entropy"],
                        max_depth=[None, 5, 7, 10],
                        min_samples_split=[5,7,10])
    elif args.section == "F":
        ### gradient boosting
        parameters = dict( n_estimators = [20,30,40,50], 
                           subsample = [0.2, 0.3, 0.4, 0.5, 0.6],
                           max_depth = [5,6,7,8,9,10])
        # parameters = dict( n_estimators = [30], 
        #                    subsample = [0.3],
        #                    max_depth = [5])
        
        
    GS = GridSearchCV(model, parameters, cv=5, verbose=verbose, n_jobs=-1)
    gs_start_time = time.perf_counter()
    GS.fit(X_train, y_train)
    gs_end_time = time.perf_counter()
    gs_time_taken_ms = (gs_end_time - gs_start_time) * 1000
    print(f"Grid Search Time Section: {args.section} Time: {gs_time_taken_ms:.4f} ms")

    if(args.section == "C" or args.section == "E"):
        print('Best Criterion:', GS.best_estimator_.get_params()['criterion'])
        print('Best Max Depth:', GS.best_estimator_.get_params()['max_depth'])
        print('Best Min Samples Split:', GS.best_estimator_.get_params()['min_samples_split'])
        if(args.section == "E"):
            print('Best Number of Estimators:', GS.best_estimator_.get_params()['n_estimators'])
    elif(args.section == "F"):
        print('Best Number of Estimators:', GS.best_estimator_.get_params()['n_estimators'])
        print('Best Max Depth:', GS.best_estimator_.get_params()['max_depth'])
        print('Best Subsample: ', GS.best_estimator_.get_params()['subsample'])
        
    model = GS.best_estimator_
    trained = False


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
print(f"Training Metrics   | Accuracy: {train_acc:.4f}% | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
print(f"Validation Metrics | Accuracy: {val_acc:.4f}% | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

### CONFUSION MATRIX
if plot_confusion:
    plot_confusion_matrix(y_true=y_train, y_pred=train_pred, labels = ["Not Face", "Face"], title=f"Train Confustion Matrix Binary Section: {args.section}")
    plot_confusion_matrix(y_true=y_val, y_pred=val_pred, labels = ["Not Face", "Face"], title=f"Validation Confusion Matrix Binary Section: {args.section}")


#### TREE VISUALIZATION
if visualization:
    plt.figure(figsize=(12,8))
    _ = tree.plot_tree(model, filled=True, fontsize=3)
    plt.show()

### TESTING
test_pred = model.predict(X_test)
test_labels = [f"{test_filename.split('.')[0]}," for test_filename in test_filenames]
df = pd.DataFrame(data={
    "labels" : test_labels,
    "pred" : test_pred.tolist()
})
if ".csv" not in args.out_path:
    args.out_path = os.path.join(args.out_path, "out.csv")
df.to_csv(args.out_path, header=False, index=False, sep=" ")



