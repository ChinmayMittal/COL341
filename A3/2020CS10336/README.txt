main_binary.py and main_multi.py are the main files.
They take the output_path and run all parts generating corresponding csv files in the output directory
USAGE:
    python main_binary.py --train_path="./train/"  --test_path="./test_sample/"  --out_path="./out/"    
    python main_multi.py --train_path="./train/"  --test_path="./test_sample/"  --out_path="./out/"

I have also submitted main_binary_ind.py and main_multi_ind.py which can be used to run individiaul parts
perform grid search and visualization etc.
USAGE:
    python main_binary_ind.py --train_path=./train/ --val_path=./validation/ --test_path=./test_sample/ --out_path=./out.csv --section=A
    python main_multi_ind.py --train_path=./train/ --val_path=./validation/ --test_path=./test_sample/ --out_path=./out.csv --section=A
The parameters at the top of scripts can be modified.

dtree.py contains my implementation of Decision Trees from scratch