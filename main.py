import argparse
import os
import pandas as pd
from utilities import read_data, plot_loss
from model import LinearRegression

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
    
    # if not os.path.isdir(args.out_path):
    #     os.makedirs(args.out_path)
        
    assert (args.section in [1,2,5])
    
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
    train_X, train_y = train_features, train_scores
    val_X, val_y = val_features, val_scores
    num_train_samples, num_features = train_X.shape
    loss = "MSE"
    learning_rate = 0.001
    stopping_criterion = "maxit"
    max_iters = 100000

    model = LinearRegression(loss=loss, num_features=num_features, 
                                        learning_rate=learning_rate, 
                                        stopping_criterion=stopping_criterion,
                                        max_iters = max_iters)
    
    ### TRAINING_LOOP
    print("Training Started ... ")
    while(True):

        model.update_weights(train_X, train_y, val_X, val_y)
        if(model.training_finished()):
            break
        
    
    print("Training Finished ... ")
    
    ### LOSS CURVES
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
    print(test_df)
    
    

    
    
    
    