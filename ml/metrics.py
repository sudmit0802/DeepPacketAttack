import multiprocessing
from pathlib import Path
from tqdm import tqdm
import datasets
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import dataset_collate_function


def confusion_matrix(data_path, model, num_class):
    data_path = Path(data_path)
    model.eval()

    cm = np.zeros((num_class, num_class), dtype=float)
    

    dataset_dict = datasets.load_dataset(str(data_path.absolute()))
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        num_workers=num_workers,
        collate_fn=dataset_collate_function,
    )
    
    print()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches")
    
    for batch_idx, batch in enumerate(dataloader):
        x = batch["feature"].float().to(model.device)
        y = batch["label"].long()
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)
        
        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1
        
        progress_bar.update(1)
        progress_bar.set_description(f'Batch {batch_idx+1}/{len(dataloader)}')
        
    progress_bar.close()
    return cm


def get_precision(cm, i):
    tp = cm[i, i]
    tp_fp = cm[:, i].sum()
    if tp_fp != 0:
        return tp / tp_fp
    else:
        return 0


def get_recall(cm, i):
    tp = cm[i, i]
    p = cm[i, :].sum()

    return tp / p


def get_classification_report(cm, labels=None):
    rows = []
    sum_prec = 0
    sum_rec = 0
    for i in range(cm.shape[0]):
        precision = get_precision(cm, i)
        if not precision:
            continue
        recall = get_recall(cm, i)
        if labels:
            label = labels[i]
        else:
            label = i

        sum_rec+=precision
        sum_prec+=recall
        row = {"label": label, "precision": precision, "recall": recall}
        rows.append(row)

    return pd.DataFrame(rows), round(sum_prec/len(rows), 4), round(sum_rec/len(rows), 4)
