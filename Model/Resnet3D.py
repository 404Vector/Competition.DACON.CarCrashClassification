import torch.nn as nn

__all__=[
    "Resnet3D_v1",
]
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super().__init__()
        # BatchNorm을 사용하므로 bias를 사용하지 않는다.
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel, (3, 3, 3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(mid_channel),
            nn.ReLU(),
            nn.Conv3d(mid_channel, out_channel, (3, 3, 3), stride=(1,1,1), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(out_channel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, (1,1,1), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(out_channel)
        )
        self.relu = nn.ReLU()
        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        out = self.blocks(x) + self.shortcut(x)
        out = self.relu(out)
        return out

class Resnet3D_v1(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.feature_extract = nn.Sequential(
            ResidualBlock3D(3,8,32),
            nn.MaxPool3d(2),
            ResidualBlock3D(32,32,64),
            nn.MaxPool3d(2),
            ResidualBlock3D(64,64,64),
            nn.MaxPool3d(2),
            ResidualBlock3D(64,64,128),
            nn.MaxPool3d((6,8,8)),
        )
        self.classifier = nn.Linear(512, num_classes)
        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
