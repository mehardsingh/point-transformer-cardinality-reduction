#!/usr/bin/env bash
device="cuda"
num_classes=40
n_epochs=50

wget http://modelnet.cs.princeton.edu/ModelNet40.zip
mv ModelNet40.zip data/modelnet40.zip
unzip data/modelnet40.zip

python -m venv env
source env/bin/activate
pip install -r requirements.txt 

git checkout -b run_models

python src/train/train_model.py --model_name "pct" --method "normal" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pct" --method "tome_ft" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pct" --method "tome_xyz" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pct" --method "random" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
    
python src/train/train_model.py --model_name "pt" --method "normal" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pt" --method "tome_ft" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pt" --method "tome_xyz" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16
python src/train/train_model.py --model_name "pt" --method "random" --dataset_name "mn40" --data_dir "data/modelnet40" --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "$device" --batch_size 16

git add outputs 
git commit -m "results"
git push --set_upstream origin run_models