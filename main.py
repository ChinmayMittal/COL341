import argparse
import os
import pandas as pd
from utilities import read_data, plot_loss
from model import LinearRegression, ScikitLearnLR, OneVsAll
from sklearn.feature_selection import SelectKBest, SelectFromModel
import sklearn.linear_model as lm

parser = argparse.ArgumentParser(
                    prog = 'Linear Regression',
                    description = 'COL341 Assignment 1',
                    epilog = 'by Chinmay Mittal')

parser.add_argument("--train_path", action="store", type=str, required=True, dest="train_path", help="path to train csv file")
parser.add_argument("--val_path", action="store", type=str, required=True, dest="val_path", help="path to val csv file")
parser.add_argument("--test_path", action="store", type=str, required=True, dest="test_path", help="path to test csv file")
parser.add_argument("--out_path", action="store", type=str, required=True, dest="out_path", help="path to generated output scores")
parser.add_argument("--section", action="store", type=int, required=True, dest="section", help="1 for Linear Regression or 2 for Ridge Regression  or 5 for Classification")
args = parser.parse_args()


if __name__ == "__main__":
    print(f"Train Path: {args.train_path}", f"Test Path: {args.test_path}", f"Val Path: {args.val_path}", f"Outut Path: {args.out_path}", sep="\n")
    
    output_dir =  os.path.split(args.out_path)[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    assert (args.section in [1,2,3,4,5,8])  ### 3 is for scikit learn implementation
    
    print(f"Section: {args.section}")
    
    print(f"Loading Data ....")
    train_sample_names, train_features, train_scores = read_data(args.train_path, test=False)
    val_sample_names, val_features, val_scores = read_data(args.val_path, test=False)
    test_sample_names, test_features, test_scores = read_data(args.test_path, test=True)
    print(f"Data Loaded .... ")
    print(f"Train Data: {train_features.shape}")
    print(f"Val Data: {val_features.shape}")
    print(f"Test Data: {test_features.shape}")
    
    ### TRAIN_SETTUP
    if not args.section == 4:
        train_X, train_y = train_features, train_scores
        val_X, val_y = val_features, val_scores
        num_train_samples, num_features = train_X.shape
        learning_rate = None
    else:
        ### FEATURE SELECTION   
        method = "select-from-model"
        
        if method == "k-best":
            
            ### K-BEST
            selector = SelectKBest(k=10)
            selector.fit(train_features, train_scores)

        else:
            
            ### SELECT FROM MODEL
            selector = SelectFromModel(estimator=lm.LinearRegression(), max_features=10)
            selector.fit(train_features, train_scores)
            
            
        train_X, train_y = selector.transform(train_features), train_scores    
        val_X, val_y = selector.transform(val_features), val_scores
        test_features, test_scores = selector.transform(test_features), test_scores
        num_train_samples, num_features = train_X.shape
        args.section = 2 ### default to ridge regression with feature selection 
        print(f"Reduced Train Data Shape: {train_X.shape}, Val Data: {val_X.shape}")
        
        learning_rate = 0.05
        
        
    if learning_rate is None:
        learning_rate = 0.001
    # stopping_criterion = "maxit" ### or "reltol"
    stopping_criterion = "reltol"
    max_iters = 10000 ### for "maxit"
    val_loss_decrease_threshold = 0.01 ### for "reltol"
    model = None
    if args.section == 1:
        print("Gradient Descent Linear Regression ... ")
        loss = "MSE"
        model = LinearRegression(loss=loss, num_features=num_features, 
                                        learning_rate=learning_rate, 
                                        stopping_criterion=stopping_criterion,
                                        max_iters = max_iters,
                                        val_loss_decrease_threshold=val_loss_decrease_threshold)
    elif args.section == 2:
        print("Ridge Regression ... ")
        loss = "ridge"
        lambda_ = 5
        model = LinearRegression(loss=loss, num_features=num_features, 
                                        learning_rate=learning_rate, 
                                        stopping_criterion=stopping_criterion,
                                        max_iters = max_iters,
                                        val_loss_decrease_threshold=val_loss_decrease_threshold,
                                        lambda_=lambda_)
    elif args.section == 3:
        print("Scikit Learn Linear Regression ... ")
        model = ScikitLearnLR()
        
    elif args.section == 8:
        print("One vs All .... ")
        max_iters = 3000
        model = OneVsAll(n_classes=9, num_features=num_features, learning_rate=learning_rate, max_iters=max_iters)
        
            
    
    ### TRAINING_LOOP
    print("Training Started ... ")
    while(True):

        model.update_weights(train_X, train_y, val_X, val_y)
        if(model.training_finished()):
            break
        
    print("Training Finished ... ")
    
    print(f"Training MSE: {model.custom_loss(train_X, train_y, loss='MSE')}, Val MSE: {model.custom_loss(val_X, val_y, loss='MSE')}")
    print(f"Training MAE: {model.custom_loss(train_X, train_y, loss='MAE')}, Val MAE: {model.custom_loss(val_X, val_y, loss='MAE')}")
    ### LOSS CURVES
    if(model.train_loss is not None):
        plot_loss(model.train_loss, model.val_loss)
    
    print("Inference on Test Set ... ")
    test_pred = model.pred(test_features)
    test_sample_names = [s_name + "," for s_name in test_sample_names]
    test_data = {
        "sample_names" : test_sample_names,
        "pred" :  test_pred.tolist() 
    }
    test_df = pd.DataFrame(data=test_data)
    test_df.to_csv(args.out_path, header=False, index=False, sep=" ")
    
    

    
    
    
    