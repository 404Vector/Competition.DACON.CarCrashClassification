import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InputConverter:
    def __init__(self, input_size:int) -> None:
        self.atf = A.Compose([
            A.Resize(input_size, input_size),
        ])

    def __call__(self, image:np.ndarray) -> np.ndarray:
        image = self.atf(image = image)['image']
        image = image / 255.
        return image
        
class ImageInputConverter:
    def __init__(self, input_size:int) -> None:
        self.atf = A.Compose([
            A.SafeRotate(limit=(-90, 90)),
            A.RandomResizedCrop(input_size, input_size, scale=(0.8,1)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __call__(self, image:np.ndarray) -> np.ndarray:
        image = self.atf(image = image)['image']
        return image