#!/usr/bin/env bash

set -e
set -x

device="cuda:2"
num_classes=40
n_epochs=30
output_dir=outputs/test/
input_dir="data/modelnet40/ModelNet40/"

mkdir -p "$output_dir"

commit_and_push() {
    git add "$output_dir"
    git commit -m "results"
    git push -u origin mehar_deep_singh
}

# wget http://modelnet.cs.princeton.edu/ModelNet40.zip
# mv ModelNet40.zip data/modelnet40.zip
# unzip data/modelnet40.zip
# mv ModelNet40/ data/modelnet40

# python3 -m venv env
source env/bin/activate
# pip install -r requirements.txt 

# git checkout -b mehar_deep_singh

# commit_and_push
# python3 src/train/train_model.py --model_name "pct" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/1/" --device "$device" --batch_size 16
# commit_and_push
# python3 src/train/train_model.py --model_name "pct" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/2/" --device "$device" --batch_size 16
# commit_and_push
# python3 src/train/train_model.py --model_name "pct" --method "tome_xyz" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/3/" --device "$device" --batch_size 16
# commit_and_push
# python3 src/train/train_model.py --model_name "pct" --method "random" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/4/" --device "$device" --batch_size 16
# commit_and_push   
# python3 src/train/train_model.py --model_name "pt" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "${output_dir}/5/" --device "$device" --batch_size 16
# commit_and_push
python3 src/train/train_model.py --model_name "pt" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "${output_dir}/6/" --device "$device" --batch_size 32
commit_and_push
# python3 src/train/train_model.py --model_name "pt" --method "tome_xyz" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "${output_dir}/7/" --device "$device" --batch_size 16
# commit_and_push
python3 src/train/train_model.py --model_name "pt" --method "random" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 32 --num_epochs $n_epochs --lr 1e-3 --wd 1e-4 --save_dir "${output_dir}/8/" --device "$device" --batch_size 32
commit_and_push

