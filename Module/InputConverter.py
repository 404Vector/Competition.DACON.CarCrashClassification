import cv2
import numpy as np
import albumentations as A

class InputConverter:
    def __init__(self, input_size:int) -> None:
        self.atf = A.Compose([
            A.Resize(input_size, input_size),
        ])

    def __call__(self, image:np.ndarray) -> np.ndarray:
        image = self.atf(image = image)['image']
        image = image / 255.
        return image
        