from point_transformer_cls import get_model, get_loss
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm 
import json
import pdb
import os
import warnings
import pandas as pd

import sys
# get dataset  
sys.path.append("src/dataset")
from modelnet40 import load_modelnet40

def get_dataloaders(val=True): 
    train_ds, test_ds = load_modelnet40("data/modelnet40", max_points=None, sampled_points=1024)

    if val:
        train_size = int(0.9 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])

        train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    
    else:
        train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)
        return train_dataloader, None, test_dataloader

def preprocess_batch(batch,device):
    batch_pointclouds = batch["pointcloud"].float()
    # batch_pointclouds = torch.transpose(batch_pointclouds, 1, 2)
    batch_labels = batch["category"]
    
    return batch_pointclouds.to(device), batch_labels.to(device)

def compute_metrics(pred, labels):
    preds = pred.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    return accuracy, precision, recall, f1

def save_function(step, model, progress_dict, save_model=False, save_dir:str="outputs/point_transformer"):
    with open(os.path.join(save_dir,"progress_dict.json"), "w") as f:
        json.dump(progress_dict, f, indent=4)

    if save_model:
        torch.save(model.state_dict(), os.path.join(save_dir), "model_step{step}.pt")
    
def do_eval(eval_dl, model, loss_fn, device): 
    model.eval()
    with torch.no_grad():
        losses = 0 
        all_preds = []
        all_labels =  []
        for batch_val in eval_dl: 
            pointclouds, labels = preprocess_batch(batch_val,device=device)
            preds_val = model(pointclouds)
            loss = loss_fn(preds_val, labels).detach()
            losses += loss.item() 

            all_preds.append(preds_val.detach())
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds) 
        all_labels = torch.cat(all_labels)
        losses /= len(all_labels) 
        accuracy, precision, recall, f1 = compute_metrics(all_preds, all_labels)

        return losses, accuracy, precision, recall, f1

def train(num_epochs:int, lr:float, wd:float, device, eval_every:int, save_logs_every:int, save_every:int, save_dir:os.PathLike='outputs/point_transformer'): 
    model = get_model().float().to(device)
    loss_fn = get_loss()

    train_dl, val_dl, test_dl = get_dataloaders()

    progess_dict = dict()

    column_names = ["step", "T_Loss", "T_Accuracy", "T_Precision", "T_Recall", "T_F1", "V_Loss", "V_Accuracy", "V_Precision", "V_Recall", "V_F1"]
    progress_df = pd.DataFrame(columns=column_names)

    os.makedirs(save_dir,exist_ok=True)


    # loop: 
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = 0 
    for epoch in range(num_epochs): 
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} progress")
        for batch in pbar: 
            progess_dict[f"step_{steps+1}"] = dict()
            optimizer.zero_grad()
            pointclouds, labels = preprocess_batch(batch, device)
            preds = model(pointclouds)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            accuracy, precision, recall, f1 = compute_metrics(preds, labels)
            # progess_dict[f"step_{steps+1}"]["train"] = {"loss": loss.item(), "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

            train_metrics = [loss.item(), accuracy, precision, recall, f1]
            # progress_df = progress_df.append(new_row, ignore_index=True)


            if not steps == 0 and steps % eval_every == 0: 
                # run eval on val set
                if val_dl is not None:
                    pbar.set_description(f'Epoch {epoch+1}/{num_epochs} progress [validating]')

                    losses, accuracy, precision, recall, f1 = do_eval(val_dl, model, loss_fn, device)
                    val_metrics = [losses, accuracy, precision, recall, f1]
                    # progess_dict[f"step_{steps+1}"]["val"] = {"loss": losses, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
                    
                    model.train()

            progress_df = progress_df.append(train_metrics + val_metrics, ignore_index=True)
            
            #save progress_dict to a file
            with open(os.path.join(save_dir,"progress_dict.json"), "w") as f:
                json.dump(progess_dict, f, indent=4)
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs} progress [acc={accuracy}]')

            steps += 1 

        # save after every epoch
        with open(os.path.join(save_dir,"progress_dict.json"), "w") as f:
            json.dump(progess_dict, f, indent=4)

        if epoch % save_every == 0: 
            save_function(steps, model, progress_dict )


def main(): 
    num_epochs = 3
    lr = 1e-3
    wd = 1e-4
    eval_every = 20
    save_every = 20

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train(num_epochs, lr, wd, device, eval_every, save_every)

if __name__ == "__main__":
    warnings.filterwarnings('ignore') 
    main()
    