import argparse
import os

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
    
    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)
        
    assert (args.section in [1,2,5])
    
    print(f"Section: {args.section}")
        
    
    