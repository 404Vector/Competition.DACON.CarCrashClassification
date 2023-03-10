from torch.utils.data import Dataset
import cv2
import os
import torch
import typing
import pandas as pd
from .Result import Result
import numpy as np

class SplitWTDataset(Dataset):
    """
    0 : "Day - Normal",
    1 : "Day - Snowy",
    2 : "Day - Rainy",
    3 : "Night - Normal",
    4 : "Night - Snowy",
    5 : "Night - Rainy",
    """
    def __init__(self, 
                 transform,
                 data:pd.DataFrame) -> None:
        self.transform = transform
        self.data = data
        self.video_path_list = data['video_path'].values
        self.label_list = data['label'].values
        
    @staticmethod
    def create_dataframe(data_path:str='./data/split_video.result', 
                         csv_path:str='./data', 
                         csv_target:str='split_video.csv') -> pd.DataFrame:
        data_frame = pd.read_csv(os.path.join(csv_path, csv_target))
        data_frame = data_frame[data_frame['crash']>0]
        data_frame['video_path'] = [os.path.join(data_path, image_name) for image_name in data_frame['image_name'].values]
        data_frame['label'] = [w+(3*t) for w,t in zip(data_frame['weather'].values, data_frame['timing'].values)]
        return data_frame

    def __getitem__(self, index):
        image = cv2.imread(self.video_path_list[index]) # type: ignore
        image = self.transform(image)
        # image = torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
        if self.label_list is not None:
            label = self.label_list[index]
            # label = torch.FloatTensor(self.one_hot_encoder[label])
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.video_path_list)