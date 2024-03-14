import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_batch(batch,device):
    batch_pointclouds = batch["pointcloud"].float()
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