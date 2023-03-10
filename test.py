import Module as M
import Model as L 
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import f1_score
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',type=int, default=128)
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--csv_target',type=str, default='train.csv')
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--mode', choices=['all','crash','ego_involve','weather','timing',])
    args = parser.parse_args()
    return args

def test(args:argparse.Namespace, test_data_frame:pd.DataFrame):
    now = args.now
    mode = args.mode
    device = args.device
    mode = args.mode
    batch_size = args.batch_size
    model_name = args.model
    csv_target = args.csv_target
    converter = M.InputConverter(input_size = args.input_size)
    test_dataset = M.CustomDataset(transform=converter, mode=mode, data=test_data_frame)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    number_of_labels = len(set(test_data_frame['label'].values))
    model = L.create_model(model_name, number_of_labels)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    test_loss = []
    test_preds, test_labels = [], []
    with torch.no_grad():
        for videos, labels in tqdm(iter(test_loader)):
            videos = videos.to(device)
            logit = model(videos)
            if csv_target == 'train.csv':
                labels = labels.to(device)
                loss = criterion(logit, labels)
                test_loss.append(loss.item())
                test_preds += logit.argmax(1).detach().cpu().numpy().tolist()
                test_labels += labels.detach().cpu().numpy().tolist()
        if csv_target == 'train.csv':
            _test_loss = np.mean(test_loss)
            _test_score = f1_score(test_labels, test_preds, average='macro')

def main(args:argparse.Namespace):
    args.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    M.SeedFixer.seed_everything(args.seed)
    df = M.CustomDataset.create_dataframe(mode=args.mode, csv_target='train.csv')
    test(args=args, test_data_frame=df)

if __name__ == '__main__':
    args = parse_args()
    main(args)