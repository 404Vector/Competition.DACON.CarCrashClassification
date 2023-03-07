import torch.nn as nn
import timm
import torch

__all__=[
    "ConvEmbeding_v1",
    "ConvEmbeding_v2",
    "ConvEmbeding_v3",
]

class ConvEmbeding_v1(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv_embeding = nn.Sequential(
            # 3 x 50 x h x w -> 3 x 10 x h x w 
            nn.Conv3d(3, 3, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
            # 3 x 10 x h x w -> 3 x 2 x h x w 
            nn.Conv3d(3, 3, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
            # 3 x 2 x h x w -> 3 x 1 x h x w 
            nn.Conv3d(3, 3, (2, 3, 3), stride=(2, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        self.conv2d = timm.create_model('efficientnet_b0', pretrained=True)
        self.conv2d.classifier = torch.nn.Identity()
        self.classifier = nn.Linear(1280, num_classes)
        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv_embeding(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv2d(x)
        x = self.classifier(x)
        return x

class ConvEmbeding_v2(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv_embeding = nn.Sequential(
            # 3 x 50 x h x w -> 3 x 10 x h x w 
            nn.Conv3d(3, 3, (5, 1, 1), stride=(5, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
            # 3 x 10 x h x w -> 3 x 2 x h x w 
            nn.Conv3d(3, 3, (5, 3, 3), stride=(5, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
            # 3 x 2 x h x w -> 3 x 1 x h x w 
            nn.Conv3d(3, 3, (2, 3, 3), stride=(2, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        self.conv2d = timm.create_model('efficientnet_b0', pretrained=True)
        self.conv2d.classifier = torch.nn.Identity()
        self.classifier = nn.Linear(1280, num_classes)
        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv_embeding(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv2d(x)
        x = self.classifier(x)
        return x

class ConvEmbeding_v3(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.conv_embeding = nn.Sequential(
            # 3 x 50 x h x w -> 3 x 5 x h x w 
            nn.Conv3d(3, 3, (10, 1, 1), stride=(10, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
            # 3 x 5 x h x w -> 3 x 2 x h x w 
            nn.Conv3d(3, 3, (5, 3, 3), stride=(1, 1, 1), padding=(0,1,1), bias=False), 
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )
        self.conv2d = timm.create_model('resnet10t', pretrained=True)
        self.conv2d.fc = torch.nn.Identity()
        self.classifier = nn.Linear(512, num_classes)
        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv_embeding(x)
        x = torch.squeeze(x, dim=2)
        x = self.conv2d(x)
        x = self.classifier(x)
        return x