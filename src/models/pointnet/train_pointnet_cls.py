from pointnet_cls import get_model, get_loss
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm 
import json
import pdb

import sys
sys.path.append("src/dataset")
from modelnet40 import load_modelnet40


def get_dataloaders(val=True): 
    train_ds, test_ds = load_modelnet40("data/modelnet40/samples", max_points=None, sampled_points=1024)

    if val:
        # train_indices, val_indices = train_test_split(np.arange(0, len(train_ds)), test_size=0.3)
        # train_ds = torch.utils.data.Subset(train_ds, train_indices)
        # val_ds = torch.utils.data.Subset(train_ds, val_indices)
        

        # train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
        # val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=True)
        # test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)

        # ========================================================

        # # Define the indices for splitting the dataset
        # dataset_size = len(train_ds)
        # split_ratio = 0.7  # 80% for the first subset, 20% for the second subset
        # split_point = int(dataset_size * split_ratio)

        # # Define samplers for each subset
        # indices = torch.randperm(dataset_size)
        # train_indices = indices[:split_point]
        # val_indices = indices[split_point:]

        # # Define data loaders for each subset
        # train_sampler = SubsetRandomSampler(train_indices)
        # train_dataloader = DataLoader(train_ds, batch_size=32, sampler=train_sampler, shuffle=True)

        # val_sampler = SubsetRandomSampler(val_indices)
        # val_dataloader = DataLoader(train_ds, batch_size=32, sampler=val_sampler, shuffle=False)

        # test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)

        # ========================================================

        train_size = int(0.99 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])

        print(len(train_ds), len(val_ds))

        train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader
    
    else:
        train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False)
        return train_dataloader, None, test_dataloader

def preprocess_batch(batch,device):
    batch_pointclouds = batch["pointcloud"].double()
    batch_pointclouds = torch.transpose(batch_pointclouds, 1, 2)
    batch_labels = batch["category"]
    
    return batch_pointclouds.to(device), batch_labels.to(device)

def compute_metrics(pred, labels):
    preds = pred.argmax(dim=1).numpy()
    labels = labels.numpy()

    # print(preds.shape)
    # print(labels.shape)

    print(preds)
    print(labels)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    return accuracy, precision, recall, f1
    
def do_eval(eval_dl, model, loss_fn, device): 
    print("eval")
    model.eval()
    with torch.no_grad():
        losses = 0 
        all_preds = []
        all_labels =  []
        for batch_val in eval_dl: 
            pointclouds, labels = preprocess_batch(batch_val,device=device)
            preds_val, trans_feats = model(pointclouds)
            loss = loss_fn(preds_val,labels,trans_feats).cpu().detach()
            # losses += len(labels) * loss 
            losses += loss.item() 

            # print(preds_val.argmax(dim=1))
            # print(labels)
            
            all_preds.append(preds_val.cpu().detach())
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds) 
        all_labels = torch.cat(all_labels)
        # print(all_preds.shape, all_labels.shape)
        losses /= len(all_labels) 
        accuracy, precision, recall, f1 = compute_metrics(all_preds, all_labels)

    return losses, accuracy, precision, recall, f1



def train(num_epochs: int = 3, lr: float = 1e-3, wd: float = 1e-4, device: str = 'cpu', eval_every: int = 20, save_every = 20) -> None: 
    
    model = get_model(normal_channel=False).double().to(device)
    loss_fn = get_loss()

    train_dl, val_dl, test_dl = get_dataloaders()

    progess_dict = dict()

    # loop: 
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = 0 
    for epoch in range(num_epochs): 
        # todo: use validation
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} progress")
        for batch in pbar: 
            print("train")
            progess_dict[f"step_{steps+1}"] = dict()
            optimizer.zero_grad()
            
            pointclouds, labels = preprocess_batch(batch,device=device)

            # print(pointclouds)
            # print(labels)
            # print(pointclouds.shape)
            # print(pointclouds.dtype)
            

            # pdb.set_trace()
            preds, trans_feat = model(pointclouds)

            loss = loss_fn(preds, labels, trans_feat)
            loss.backward()
            optimizer.step()

            accuracy, precision, recall, f1 = compute_metrics(preds, labels)
            progess_dict[f"step_{steps+1}"]["train"] = {"loss": loss.item(), "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


            if not steps == 0 and steps % eval_every == 0: 
                # run eval on val set
                if val_dl is not None:
                    pbar.set_description(f'Epoch {epoch+1}/{num_epochs} progress [validating]')

                    losses, accuracy, precision, recall, f1 = do_eval(val_dl, model, loss_fn, device)
                    progess_dict[f"step_{steps+1}"]["val"] = {"loss": losses, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
                    # print({"loss": losses, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
                    model.train()
            
            if steps % save_every == 0 or int(len(train_dl) * epoch)-1 == steps:
                pbar.set_description(f'Epoch {epoch+1}/{num_epochs} progress: SAVING')
                #save progress_dict to a file
                with open("progress_dict.json", "w") as f:
                    json.dump(progess_dict, f, indent=4)
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs} progress [acc={accuracy}]')

            steps += 1 


    

# def test_model_params():
#     train_ds, test_ds = load_modelnet40("data/modelnet40/samples", max_points=None, sampled_points=1024)
#     train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
#     test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=True)

#     train_batch = next(iter(train_dataloader))
#     print(train_batch["pointcloud"].shape)

#     model = get_model(normal_channel=False).double()

#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of parameters: {num_params}")

#     sample1 = train_ds[0]
#     print(sample1)
#     pc1 = sample1["pointcloud"]
#     print(pc1.shape)
#     pointcloud = torch.cat([pc1.unsqueeze(0) for _ in range(100)], dim=0)
#     print(pointcloud.shape)


#     pointcloud = torch.transpose(pointcloud, 1, 2)
#     pointcloud = pointcloud.double()

#     print(pointcloud.shape)

#     y_hat, _ = model(pointcloud)
#     print(y_hat.shape)

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device=device, save_every=1, eval_every=2)

if __name__ == "__main__":
    main()
    