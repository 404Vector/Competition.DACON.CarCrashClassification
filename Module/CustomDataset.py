from torch.utils.data import Dataset
import cv2
import os
import torch
import typing
import pandas as pd
from .Result import Result
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, 
                 transform,
                 mode:typing.Literal['all','crash','ego_involve','weather','timing','w&t'],
                 data:pd.DataFrame) -> None:
        self.mode = mode
        self.one_hot_encoder = Result.one_hot_encoder[mode]
        self.transform = transform
        self.data = data
        self.video_path_list = data['video_path']
        self.label_list = data['label']
        
    @staticmethod
    def create_dataframe(mode:typing.Literal['all','crash','ego_involve','weather','timing','w&t']='crash',
                         data_path:str='./data', 
                         csv_target:typing.Literal['train.csv','test.csv']='train.csv') -> pd.DataFrame:
        df = pd.read_csv(os.path.join(data_path, csv_target))
        video_path_list = CustomDataset.__get_video_paths__(mode, df, data_path)
        label_list = CustomDataset.__get_label_list__(mode, df)
        df = pd.DataFrame()
        df['video_path'] = video_path_list
        if csv_target == 'train.csv':
            df['label'] = label_list
        else:
            df['label'] = [None for p in video_path_list]
        return df
    
    @staticmethod
    def __get_video_paths__(mode:str, df:pd.DataFrame, data_path:str) -> typing.List[str]:
        if mode == 'all':
            return [os.path.join(data_path, path[2:]) for path in df['video_path'].values]
        elif mode == 'crash':
            return [os.path.join(data_path, path[2:]) for path in df['video_path'].values]
        else:
            df_sub = df[df['label'] > 0]
            return [os.path.join(data_path, path[2:]) for path in df_sub['video_path'].values]
    
    @staticmethod
    def __get_label_list__(mode:str, df:pd.DataFrame) -> typing.List[int]:
        if mode == 'all':
            return [label for label in df['label'].values]
        elif mode == 'crash':
            results = [Result(label) for label in df['label'].values]
            return [result.encoded_crash for result in results]
        else:
            df_sub = df[df['label'] > 0]
            results = [Result(label) for label in df_sub['label'].values]
            if mode == 'ego_involve':
                return [result.encoded_ego_involve for result in results]
            elif mode == 'weather':
                return [result.encoded_weather for result in results]
            elif mode == 'timing':
                return [result.encoded_timing for result in results]
            elif mode == 'w&t':
                return [result.encoded_weather_by_timing for result in results]
            else:
                raise Exception(f"Unknown mode : {mode}")

    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])
        label = self.label_list[index]
        if self.label_list is not None:
            return frames, label
        else:
            return frames
        
    def __len__(self):
        return len(self.video_path_list)
    
    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(frame_count):
            _, img = cap.read()
            img = self.transform(img)
            frames.append(img)
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)