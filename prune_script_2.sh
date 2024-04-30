1#!/usr/bin/env bash

set -e
set -x

device=$1 # "cuda:2"
num_classes=40
n_epochs=20
output_dir=outputs/futher_prune_v2_0/
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

# python -m venv env
source env/bin/activate
# pip install -r requirements.txt 

# git checkout -b mehar_deep_singh

# commit_and_push
# python src/train/train_model.py --model_name "pct" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 128 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pct_no_ds/" --device "$device" --batch_size 16 --tome_further_ds 1.0 --tome_further_ds_use_xyz false
# commit_and_push
python src/train/train_model.py --model_name "pct" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 128 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pct_ds_normal/" --device "$device" --batch_size 32 --tome_further_ds 0.85 --tome_further_ds_use_xyz false
commit_and_push
python src/train/train_model.py --model_name "pct" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 128 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pct_ds_tome/" --device "$device" --batch_size 32 --tome_further_ds 0.85 --tome_further_ds_use_xyz false
commit_and_push

# python src/train/train_model.py --model_name "pt" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 42 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pt_no_ds/" --device "$device" --batch_size 16 --tome_further_ds_use_xyz false
# commit_and_push
python src/train/train_model.py --model_name "pt" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 42 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pt_ds_normal/" --device "$device" --batch_size 32 --tome_further_ds 0.85 --tome_further_ds_use_xyz false
commit_and_push
python src/train/train_model.py --model_name "pt" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 42 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pt_ds_tome/" --device "$device" --batch_size 32 --tome_further_ds 0.85 --tome_further_ds_use_xyz false
commit_and_push


python src/train/train_model.py --model_name "pct" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 128 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pct_ds_normal_more/" --device "$device" --batch_size 32 --tome_further_ds 0.7 --tome_further_ds_use_xyz false
commit_and_push
python src/train/train_model.py --model_name "pct" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 128 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pct_ds_tome_more/" --device "$device" --batch_size 32 --tome_further_ds 0.7 --tome_further_ds_use_xyz false
commit_and_push

commit_and_push
python src/train/train_model.py --model_name "pt" --method "normal" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 42 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pt_ds_normal_more/" --device "$device" --batch_size 32 --tome_further_ds 0.7 --tome_further_ds_use_xyz false
commit_and_push
python src/train/train_model.py --model_name "pt" --method "tome_ft" --dataset_name "mn40" --data_dir $input_dir --val "True" --num_classes $num_classes --num_points 1024 --k 16 --input_dim 3 --init_hidden_dim 42 --num_epochs $n_epochs --lr 1e-4 --wd 1e-4 --save_dir "${output_dir}/pt_ds_tome_more/" --device "$device" --batch_size 32 --tome_further_ds 0.7 --tome_further_ds_use_xyz false
commit_and_push


