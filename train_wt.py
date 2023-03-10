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
import timm
from segmentation_models_pytorch.losses import FocalLoss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int, default=21)
    parser.add_argument('--input_size',type=int, default=128)
    parser.add_argument('--batch_size',type=int, default=16)
    parser.add_argument('--epoch',type=int, default=10)
    parser.add_argument('--lr',type=float, default=1e-3)
    parser.add_argument('--model',type=str, default='efficientnet_b0')
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--mode', type=str, default='w&t')
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

    converter = M.ImageInputConverter(input_size = args.input_size)
    train_dataset = M.SplitWTDataset(transform=converter, data=train_data_frame)
    valid_dataset = M.SplitWTDataset(transform=converter, data=valid_data_frame)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=number_of_workers)
    
    # AMP : loss scale์ ์ํ GradScaler ์์ฑ
    scaler = GradScaler()
    number_of_labels = len(set(train_data_frame['label'].values))
    model = timm.create_model(model_name=model_name, num_classes=number_of_labels, pretrained=True) #L.create_model(model_name, number_of_labels)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = FocalLoss(mode='multiclass', alpha=0.25).to(device)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    best_val_score = 0

    for epoch in range(1, epoch+1):
        model.train()
        train_loss = []
        train_preds, train_labels = [], []
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
            train_preds += output.argmax(1).detach().cpu().numpy().tolist()
            train_labels += labels.detach().cpu().numpy().tolist()
        _train_score = f1_score(train_labels, train_preds, average='macro')
        _train_loss = np.mean(train_loss)
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
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Train F1 : [{_train_score:.5f}]')
        M.print_each_acc(M.get_each_acc(number_of_labels, train_labels, train_preds))
        print(f'Epoch [{epoch}], Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
        M.print_each_acc(M.get_each_acc(number_of_labels, val_labels, val_preds))
        wandb.log({
            "Train_loss": _train_loss,
            "Train_F1": _train_score,
            "Valid_loss": _val_loss,
            "Valid_F1": _val_score,
        })
        if best_val_score < _val_score:
            best_val_score = _val_score
            time = now
            torch.save(model.state_dict(), os.path.join(data_path, f'{time}-{mode}_model.pkl'))

def main(args:argparse.Namespace):
    args.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    M.SeedFixer.seed_everything(args.seed)
    df = M.SplitWTDataset.create_dataframe()
    skf = StratifiedKFold(shuffle=True, random_state=args.seed)
    df_0 = pd.DataFrame(df[df['frame_idx'] == 0].to_dict(orient='list'))
    splited_iters = skf.split(df_0['sample_id'], df_0['label'])
    train_idxs, valid_idxs = next(splited_iters)
    train_iloc_key = df_0.iloc[train_idxs]['sample_id'].values
    valid_iloc_key = df_0.iloc[valid_idxs]['sample_id'].values
    train_idxs_full = [sample_id in train_iloc_key for sample_id in df['sample_id'].values]
    valid_idxs_full = [sample_id in valid_iloc_key for sample_id in df['sample_id'].values]
    train_df = pd.DataFrame(df_0.iloc[train_idxs].to_dict(orient='list'))
    valid_df = pd.DataFrame(df_0.iloc[valid_idxs].to_dict(orient='list'))
    wandb.init(
        project="Competition.DACON.CarCrashClassification",
        group=args.mode,
        name=f"{args.mode}.{args.model}",
        config=args,
    )
    train(args=args, train_data_frame=train_df, valid_data_frame=valid_df)
    wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    main(args)