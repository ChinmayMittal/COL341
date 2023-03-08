python main.py --help => for help in running main.py and description of arguments

python main.py --train_path ./data/train.csv \
               --val_path ./data/val.csv \
               --test_path ./data/test.csv \
               --out_path ./output/out.csv \ 
               --lr 0.001 \ 
               --sc maxit  \ combined|maxit|reltol
               --maxit 1000 \  
               --valdec 0.01 \ 
               --lambda 5 \ => for ridge regression
               --section 1

other arguments can be set to default
python main.py --train_path ./data/train.csv \
               --val_path ./data/validation.csv \
               --test_path ./data/test.csv \
               --out_path ./output/out.csv \
               --section 1