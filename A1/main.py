import argparse
import os
import numpy as np
import pandas as pd
import pickle
from utilities import read_data, plot_loss
from model import LinearRegression, ScikitLearnLR, OneVsAll, LinearClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
                    prog = 'Linear Regression',
                    description = 'COL341 Assignment 1',
                    epilog = 'by Chinmay Mittal')

parser.add_argument("--train_path", action="store", type=str, required=True, dest="train_path", help="path to train csv file")
parser.add_argument("--val_path", action="store", type=str, required=True, dest="val_path", help="path to val csv file")
parser.add_argument("--test_path", action="store", type=str, required=True, dest="test_path", help="path to test csv file")
parser.add_argument("--out_path", action="store", type=str, required=True, dest="out_path", help="path to generated output scores")
parser.add_argument("--lr", action="store", type=float, required=False, dest="learning_rate", help="learning_rate", default=0.0005)
parser.add_argument("--sc", action="store", type=str, required=False, dest="stopping_criterion", help="stopping_criterion maxit|reltol|combined", default="combined")
parser.add_argument("--maxiter", action="store", type=int, required=False, dest="maxiter", help="max number of iterations", default=500)
parser.add_argument("--valdec", action="store", type=float, required=False, dest="val_loss_decrease_threshold", help="val decrease threshold", default=0.00001)
parser.add_argument("--lambda", action="store", type=float, required=False, dest="lambda_", help="Ridge Regularizer", default=5)
parser.add_argument("--section", action="store", type=int, required=True, dest="section", help="1 for Linear Regression or 2 for Ridge Regression  or 5 for Classification and 3 For Scikit Learn 4 for Feature selection and 8 for Bonus")
args = parser.parse_args()


if __name__ == "__main__":
    print(f"Train Path: {args.train_path}", f"Test Path: {args.test_path}", f"Val Path: {args.val_path}", f"Outut Path: {args.out_path}", sep="\n")
    
    output_dir =  os.path.split(args.out_path)[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    assert (args.section in [1,2,3,4,5,8])  ### 3 is for scikit learn implementation
    
    print(f"Section: {args.section}")
    
    ### for generalizaton analysis data is in different format
    generalization_analysis = False
    print(f"Loading Data ....")
    train_sample_names, train_features, train_scores = read_data(args.train_path, test=False, scores_last=generalization_analysis)
    val_sample_names, val_features, val_scores = read_data(args.val_path, test=False, scores_last=generalization_analysis)
    test_sample_names, test_features, test_scores = read_data(args.test_path, test=True)
    print(f"Data Loaded .... ")
    

    ### FOR DATA SPLIT EXPERIMENT
    data_split = 1
    if data_split != 1:
        train_features, second_train_features, train_scores, second_train_scores= train_test_split(train_features, train_scores, test_size=(1-data_split), random_state=1)
    # train_features, train_scores = second_train_features, second_train_scores
    
    print(f"Train Data: {train_features.shape}")
    print(f"Val Data: {val_features.shape}")
    print(f"Test Data: {test_features.shape}")
    
    
    #### DATA NORMALIZATION SETTUP
    normalization = False
    if normalization:
        mean = np.mean(train_features, axis=0)
        variance = np.std(train_features, axis=0)
        train_features  = (train_features-mean)/variance
        test_features = (test_features-mean)/variance
        val_features =(val_features-mean)/variance   
        print(train_features.shape)     
    ### TRAIN_SETTUP
    
    if not args.section == 4:
        train_X, train_y = train_features, train_scores
        val_X, val_y = val_features, val_scores
        num_train_samples, num_features = train_X.shape
        learning_rate = None
    else:
        ### FEATURE SELECTION   
        method = "select-from-model"
        # method = "k-best"
        num_features_selected  = 10
        if method == "k-best":
            
            ### K-BEST
            selector = SelectKBest(k=num_features_selected)
            selector.fit(train_features, train_scores)

        else:
            
            ### SELECT FROM MODEL
            selector = SelectFromModel(estimator=lm.LinearRegression(), max_features=num_features_selected)
            selector.fit(train_features, train_scores)
            
            
        train_X, train_y = selector.transform(train_features), train_scores    
        val_X, val_y = selector.transform(val_features), val_scores
        test_features, test_scores = selector.transform(test_features), test_scores
        num_train_samples, num_features = train_X.shape
        args.section = 2 ### default to ridge regression with feature selection 
        print(f"Reduced Train Data Shape: {train_X.shape}, Val Data: {val_X.shape}")
        
        learning_rate = 0.001
        
        
    if learning_rate is None:
        learning_rate = args.learning_rate

    stopping_criterion = args.stopping_criterion
    max_iters = args.maxiter ### for "maxit"
    val_loss_decrease_threshold = args.val_loss_decrease_threshold ### for "reltol"
    model = None
    print(f"Learning Rate: {learning_rate}, SC: {stopping_criterion}, Max Iters: {max_iters}, Val Decrease: {val_loss_decrease_threshold}")
    if args.section == 1:
        print("Gradient Descent Linear Regression ... ")
        loss = "MSE"
        model = LinearRegression(loss=loss, num_features=num_features, 
                                        learning_rate=learning_rate, 
                                        stopping_criterion=stopping_criterion,
                                        max_iters = max_iters,
                                        val_loss_decrease_threshold=val_loss_decrease_threshold)
    elif args.section == 2:
        loss = "ridge"
        lambda_ = args.lambda_
        print(f"Ridge Regression ... lambda: {lambda_}")
        model = LinearRegression(loss=loss, num_features=num_features, 
                                        learning_rate=learning_rate, 
                                        stopping_criterion=stopping_criterion,
                                        max_iters = max_iters,
                                        val_loss_decrease_threshold=val_loss_decrease_threshold,
                                        lambda_=lambda_)
    elif args.section == 3:
        print("Scikit Learn Linear Regression ... ")
        model = ScikitLearnLR()
    
    elif args.section == 5:
        print(" Linear Classification ... ")
        model = LinearClassifier(n_classes=9, num_features=num_features, learning_rate=learning_rate, max_iters=max_iters, val_loss_decrease_threshold=val_loss_decrease_threshold, stopping_criterion=stopping_criterion)
    elif args.section == 8:
        print("One vs All .... ")
        max_iters = 3000
        model = OneVsAll(n_classes=9, num_features=num_features, learning_rate=learning_rate, max_iters=max_iters)
        
            
    
    ### TRAINING_LOOP
    print("Training Started ... ")
    while(True):
        # print(f"Num Updates: {model.num_updates}", end="\r")
        model.update_weights(train_X, train_y, val_X, val_y)
        if(model.training_finished()):
            break
    print("Training Finished ... ")
    
    print(f"Training MSE: {model.custom_loss(train_X, train_y, loss='MSE')}, Val MSE: {model.custom_loss(val_X, val_y, loss='MSE')}")
    print(f"Training MAE: {model.custom_loss(train_X, train_y, loss='MAE')}, Val MAE: {model.custom_loss(val_X, val_y, loss='MAE')}")
    
    plotting = True
    ### LOSS CURVES
    if(model.train_loss is not None and plotting):
        label = "Log Loss" if (args.section == 5 or args.section == 8 ) else "MSE"
        shift = False if args.section == 5 or generalization_analysis else True
        # class_loss = model.class_loss
        class_loss = None
        plot_loss(model.train_loss, model.val_loss, class_loss = class_loss, label = label, shift = shift)
    
    if(args.section == 5 or args.section == 8):
        print(f"Train Accuracy: {model.accuracy(train_features, train_scores)}")
        print(f"Val Accuracy: {model.accuracy(val_features, val_scores)}")

    if not generalization_analysis:
        print("Inference on Test Set ... ")
        test_pred = model.pred(test_features)
        test_sample_names = [s_name + "," for s_name in test_sample_names]
        test_data = {
            "sample_names" : test_sample_names,
            "pred" :  test_pred.tolist() 
        }
        test_df = pd.DataFrame(data=test_data)
        if ".csv" not in args.out_path:
            args.out_path = os.path.join(args.out_path, "out.csv")
        test_df.to_csv(args.out_path, header=False, index=False, sep=" ")
    
    
    # num_points = {}
    # for class_idx in range(1,10):
    #     num_points[class_idx] = np.sum((train_y==class_idx).astype(int))
    #     print(num_points[class_idx], class_idx)
    # plt.scatter(class_idx, model.class_loss[class_idx][-1]*num_points[class_idx], marker="X", color="red")
    # plt.xlabel("Class")
    # plt.ylabel("Total Loss")
    # plt.title("Class wise Total Loss")
    # plt.grid(True)
    # plt.show()
    
    # plt.xlabel("Number of Features")
    # plt.ylabel("Average MSE")
    # plt.title("Loss vs Number of Features")
    # plt.scatter((10,100,1000,2048), (2.28, 1.03, 0.477, 0.477), marker="X", color="red", label="Train MSE")
    # plt.scatter((10,100,1000,2048), (1.99, 1.1, 0.903, 0.903), marker="X", color="orange", label="Val MSE")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # plt.xlabel("D")
    # plt.ylabel("|E_out - E_in |")
    # plt.title("Generalization Analysis")
    # plt.scatter((2,5,10,100), (0.090, 0.114, 0.32, 1.461), marker="X", color="red")
    # plt.grid(True)
    # plt.show()
        

    # file = open('part-2-data-1', 'wb')

    # # dump information to that file
    # pickle.dump(model, file)

    # # close the file
    # file.close()
    
    # file = open('part-2-data-1', 'rb')

    # # dump information to that file
    # model1 = pickle.load(file)
    # pred1 = model1.pred(val_features)
    # # close the file
    # file.close()
    
    # file = open('part-2-data-2', 'rb')

    # # dump information to that file
    # model2 = pickle.load(file)
    # pred2 = model2.pred(val_features)
    # # close the file
    # file.close()
    
    # print(np.mean(np.abs(pred1-pred2)))
    
    
    
    
    
    