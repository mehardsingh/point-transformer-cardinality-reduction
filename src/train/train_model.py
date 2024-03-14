import torch
import numpy as np
from tqdm import tqdm 
import os
import warnings
from train_utils import preprocess_batch, compute_metrics, do_eval
import argparse
import sys
from modelnet40 import get_dataloaders as get_mn40_dls
from config import Config as ModelConfig

sys.path.append("src/models/pct")
from point_transformer_cls import get_model as get_pct, get_loss as get_pct_loss

sys.path.append("src/models/pt")
from model import get_model as get_pt, get_loss as get_pt_loss

sys.path.append("src/downsamples/fps_knn_pct")
from fps_knn_pct import FPS_KNN_PCT

sys.path.append("src/tome")
from tome import TOME

def get_model(model_name, tome, num_points, num_class, input_dim, init_hidden_dim, k, device):
    model_config = ModelConfig(tome, num_points, num_class, input_dim, init_hidden_dim, k)
    if model_name == "pct":
        model = get_pct(model_config).float().to(device)
        loss_fn = get_pct_loss()
    elif model_name == "pt":
        model = get_pt(model_config).float().to(device)
        loss_fn = get_pt_loss()
    else:
        raise ValueError(f"The provided model_name is not supported: {model_name}")
    
    return model, loss_fn

def get_dataloaders(dataset_name, data_dir, num_points, val, k):
    if dataset_name == "mn40":
        train_dl, val_dl, test_dl = get_mn40_dls(data_dir, num_points, val, k)
    else:
        raise ValueError(f"Bad dataset provided")
    
    eval_dl = val_dl if val else test_dl
    return train_dl, eval_dl

def initialize_progress_csv(save_dir):
    column_names = ["step", "T_Loss", "T_Accuracy", "T_Precision", "T_Recall", "T_F1", "V_Loss", "V_Accuracy", "V_Precision", "V_Recall", "V_F1"]
    with open(os.path.join(save_dir, "progress.csv"), mode="w") as f:
        f.write(f"{','.join(column_names)}\n")

def save_progress(save_dir, steps, train_metrics, eval_metrics, model):
    with open(os.path.join(save_dir, "progress.csv"), mode="a") as f:
        row_metrics = [steps] + train_metrics + eval_metrics
        row_metrics = [str(i) for i in row_metrics]
        f.write(f"{','.join(row_metrics)}\n")

    torch.save(model.state_dict(), os.path.join(save_dir, f"model_step{steps}.pt"))

def get_downsample(downsample_name):
    if downsample_name == "fps_knn":
        downsample = FPS_KNN_PCT
    elif downsample_name == "tome":
        downsample = TOME
    else:
        raise ValueError(f"Invalid downsample name: {downsample_name}")
    
    return downsample

def train(config): 
    if not config["save_dir"]:
        raise ValueError("The save_dir is None")
    else:
        os.makedirs(config["save_dir"], exist_ok=True)

    model, loss_fn = get_model(
        config["model_name"], 
        config["tome"], 
        config["num_points"],
        config["num_classes"],
        config["input_dim"],
        config["init_hidden_dim"],
        config["k"],  
        config["device"]
    )

    train_dl, eval_dl = get_dataloaders(
        config["dataset_name"],
        config["data_dir"],
        config["num_points"],
        config["val"],
        config["num_classes"]
    )

    initialize_progress_csv(config["save_dir"])

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])

    curr_step = 1
    for epoch in range(config["num_epochs"]): 
        pbar = tqdm(train_dl, desc="Epoch {}/{} progress".format(epoch+1, config["num_epochs"]))
        batch_train_metrics = list()

        for batch in pbar: 
            optimizer.zero_grad()
            pointclouds, labels = preprocess_batch(batch, config["device"])
            preds = model(pointclouds)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(preds, labels)
            train_metrics = [loss.item(), train_accuracy, train_precision, train_recall, train_f1]
            batch_train_metrics.append(train_metrics)

            pbar.set_description("Epoch {}/{} progress [train_acc={}]".format(epoch+1, config["num_epochs"], train_accuracy))
            curr_step += 1 

        avg_train_metrics = list(np.mean(np.array(batch_train_metrics), axis=0))

        pbar.set_description("Epoch {}/{} progress [evaluating]".format(epoch+1, config["num_epochs"]))
        eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = do_eval(eval_dl, model, loss_fn, config["device"])
        eval_metrics = [eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1]                    
        model.train()

        save_progress(config["save_dir"], curr_step, avg_train_metrics, eval_metrics, model)

def main(args): 
    config = vars(args)
    config["tome"] = True if config["tome"] == "True" else False
    config["val"] = True if config["val"] == "True" else False
    train(config)

if __name__ == "__main__":
    warnings.filterwarnings('ignore') 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="pt")
    parser.add_argument("--tome",  type=str, default="False")
    parser.add_argument("--dataset_name",  type=str, default="mn40")
    parser.add_argument("--data_dir",  type=str, default="data/modelnet40")
    parser.add_argument("--val",  type=str, default="False")
    parser.add_argument("--num_classes",  type=int, default=40)
    parser.add_argument("--num_points",  type=int, default=1024)
    parser.add_argument("--k",  type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--init_hidden_dim", type=int, default=64)
    parser.add_argument("--num_epochs",  type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    main(args)
    
# python src/train/train_model.py --model_name "pct" --tome "True" --dataset_name "mn40" --data_dir "data/modelnet40" --val "False" --num_classes 10 --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs 10 --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "cpu"
# python src/train/train_model.py --model_name "pt" --tome "False" --dataset_name "mn40" --data_dir "data/modelnet40" --val "False" --num_classes 10 --num_points 1024 --k 32 --input_dim 3 --init_hidden_dim 64 --num_epochs 10 --lr 1e-3 --wd 1e-4 --save_dir "outputs/pt_tome" --device "cpu"
