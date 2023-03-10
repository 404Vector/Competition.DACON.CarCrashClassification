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
import timm
import cv2

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',type=int, default=128)
    parser.add_argument('--batch_size',type=int, default=1)
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--csv_target',type=str, default='train.csv')
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--mode',type=str, default='w&t')
    parser.add_argument('--model',type=str, default='efficientnet_b0')
    parser.add_argument('--weight',type=str, default='2023-03-09-13-45-10-w&t_model(efb0).pkl')
    args = parser.parse_args()
    return args

def test(args:argparse.Namespace, test_data_frame:pd.DataFrame):
    now = args.now
    mode = args.mode
    device = args.device
    data_path = args.data_path
    batch_size = args.batch_size
    model_name = args.model
    csv_target = args.csv_target
    weight_name = args.weight
    weight_path = os.path.join(data_path, weight_name)
    converter = M.InputConverter(input_size = args.input_size)
    test_dataset = M.CustomDataset(transform=converter, mode=mode, data=test_data_frame)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    number_of_labels = len(set(test_data_frame['label'].values))
    model = timm.create_model(model_name=model_name, num_classes=6, pretrained=False)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))
    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()
    test_loss = []
    test_preds, test_labels = [], []
    decode_map = {
        0:'day-normal',
        1:'day-snowy',
        2:'day-rainy',
        3:'night-normal',
        4:'night-snowy',
        5:'night-rainy',
    }
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(tqdm(iter(test_loader))):
            images = videos[:,:,0,:,:]
            images = images.to(device)
            logit = model(images)
            pred = logit.argmax(1).detach().cpu().numpy().tolist()
            for idx, (_img, _pred) in enumerate(zip(videos[:,:,0,:,:], pred)):
                _img = _img.permute(1,2,0).numpy()*255
                cv2.imwrite(f'./data/temp/{(batch_size*batch_idx) + idx}-{decode_map[_pred]}.jpg', _img)

def main(args:argparse.Namespace):
    args.now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    df = M.CustomDataset.create_dataframe(mode=args.mode, csv_target='train.csv')
    test(args=args, test_data_frame=df)

if __name__ == '__main__':
    args = parse_args()
    main(args)