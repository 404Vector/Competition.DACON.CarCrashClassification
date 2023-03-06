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
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int, default=21)
    parser.add_argument('--input_size',type=int, default=128)
    parser.add_argument('--batch_size',type=int, default=5)
    parser.add_argument('--epoch',type=int, default=10)
    parser.add_argument('--lr',type=float, default=1e-3)
    parser.add_argument('--model',type=str, default='Resnet3D_v1')
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--mode', choices=['all','crash','ego_involve','weather','timing',])
    parser.add_argument('--number_of_workers',type=int, default=0)
    args = parser.parse_args()
    return args

def train(args:argparse.Namespace, train_data_frame:pd.DataFrame, valid_data_frame:pd.DataFrame):
    now = args.now
    device = args.device
    epoch = args.epoch
    data_path = args.data_path
    lr = args.lr
    mode = args.mode
    number_of_workers = args.number_of_workers
    batch_size = args.batch_size
    model_name = args.model

    converter = M.InputConverter(input_size = args.input_size)
    train_dataset = M.CustomDataset(transform=converter, mode=mode, data=train_data_frame)
    valid_dataset = M.CustomDataset(transform=converter, mode=mode, data=valid_data_frame)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=number_of_workers)
    
    # AMP : loss scale을 위한 GradScaler 생성
    scaler = GradScaler()
    number_of_labels = len(set(train_data_frame['label'].values))
    model = L.create_model(model_name, number_of_labels)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = M.FocalLoss().to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    best_val_score = 0

    for epoch in range(1, epoch+1):
        model.train()
        train_loss = []
        # Train
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(videos)
                loss = criterion(output, labels)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward() # loss.backward()
            scaler.step(optimizer) # optimizer.step()
            scaler.update() # Updates the scale for next iteration.
            train_loss.append(loss.item())
        # Valid
        model.eval()
        val_loss = []
        val_preds, val_labels = [], []
        with torch.no_grad():
            for videos, labels in tqdm(iter(valid_loader)):
                videos = videos.to(device)
                labels = labels.to(device)
                logit = model(videos)
                loss = criterion(logit, labels)
                val_loss.append(loss.item())
                val_preds += logit.argmax(1).detach().cpu().numpy().tolist()
                val_labels += labels.detach().cpu().numpy().tolist()

            _val_loss = np.mean(val_loss)

        _val_score = f1_score(val_labels, val_preds, average='macro')

        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        wandb.log({
            "Train_loss": _train_loss,
            "Valid_loss": _val_loss,
            "Valid_F1": _val_score,
        })
        if best_val_score < _val_score:
            best_val_score = _val_score
            time = now
            torch.save(model.state_dict(), os.path.join(data_path, f'{time}-{mode}_model.pkl'))

def main(args:argparse.Namespace):
    args.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    wandb.init(
        project="Competition.DACON.CarCrashClassification",
        group=args.mode,
        name=f"{args.mode}.{args.model}",
        config=args,
    )
    M.SeedFixer.seed_everything(args.seed)
    df = M.CustomDataset.create_dataframe(mode=args.mode, csv_target='train.csv')
    skf = StratifiedKFold(shuffle=True, random_state=args.seed)
    splited_iters = skf.split(df['video_path'], df['label'])
    
    train_idxs, valid_idxs = next(splited_iters)
    train_df = pd.DataFrame(df.iloc[train_idxs].to_dict(orient='list'))
    valid_df = pd.DataFrame(df.iloc[valid_idxs].to_dict(orient='list'))
    train(args=args, train_data_frame=train_df, valid_data_frame=valid_df)
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)