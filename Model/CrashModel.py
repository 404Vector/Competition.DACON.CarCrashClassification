import torch
import torch.nn as nn
import pytorchvideo.models as pv

__all__=[
    "CrashModel_v1",
    "CrashModel_v2",
]

class CrashModel_v1(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((3, 7, 7)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

class CrashModel_v2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((3, 7, 7)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x