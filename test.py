import Module as M
import Model as L 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import f1_score
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['w&t','c&e'])
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--batch_size_ce',type=int, default=8)
    parser.add_argument('--input_size_ce',type=int, default=128)
    parser.add_argument('--model_ce',type=str, default='Resnet3D_v2')
    parser.add_argument('--weight_ce',type=str, default='[best c&e]2023-03-11-10-13-34-c&e_model.pkl')
    parser.add_argument('--batch_size_wt',type=int, default=8)
    parser.add_argument('--input_size_wt',type=int, default=512)
    parser.add_argument('--model_wt',type=str, default='efficientnet_b4')
    parser.add_argument('--weight_wt',type=str, default='[best w&t]2023-03-10-19-32-30-w&t_model.pkl')
    args = parser.parse_args()
    return args

def test_ce(args:argparse.Namespace, test_data_frame:pd.DataFrame):
    device = args.device
    data_path = args.data_path
    batch_size = args.batch_size_ce
    model_name = args.model_ce
    weight_name = args.weight_ce
    input_size = args.input_size_ce 

    converter = M.InputConverter(input_size = input_size)
    test_dataset = M.CustomDataset(transform=converter, mode='c&e', data=test_data_frame)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = L.create_model(model_name, 3)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(data_path, weight_name)))
    model.eval()
    test_preds = []

    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)
            logit = model(videos)
            test_preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return test_preds

def test_wt(args:argparse.Namespace, test_data_frame:pd.DataFrame):
    device = args.device
    data_path = args.data_path
    batch_size = args.batch_size_wt
    model_name = args.model_wt
    weight_name = args.weight_wt
    input_size = args.input_size_wt 

    converter = M.InputConverter(input_size = input_size)
    test_dataset = M.CustomDataset(transform=converter, mode='w&t', data=test_data_frame)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = timm.create_model(model_name=model_name, num_classes=6)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(data_path, weight_name)))
    model.eval()
    test_preds = []

    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            images = videos[:,:,0,:,:]
            images = images.to(device)
            logit = model(images)
            test_preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return test_preds

def main(args:argparse.Namespace):
    if args.mode == 'c&e':
        df = M.CustomDataset.create_dataframe(mode="c&e", csv_target='test.csv')
        preds = test_ce(args=args, test_data_frame=df)
        result_name = f"{args.weight_ce}.csv"
    elif args.mode == 'w&t':
        df = M.CustomDataset.create_dataframe(mode="w&t", csv_target='test.csv')
        preds = test_wt(args=args, test_data_frame=df)
        result_name = f"{args.weight_wt}.csv"
    else:
        raise Exception(f'ERROR! {args.mode} is wrong mode.')
    submission = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'))
    submission['label'] = preds
    submission.to_csv(os.path.join(args.data_path, result_name), index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)