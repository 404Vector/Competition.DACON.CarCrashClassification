import torch
import torch.nn as nn
import pytorchvideo.models as pv
import timm

__all__=[
    "WTModel_v1",
]

class WTModel_v1(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # backborns = [timm.create_model('efficientnet_b0', num_classes=num_classes, pretrained=True) for _ in range(50)]
        self.backborn = pv.resnet.create_resnet(
            input_channel=3, # RGB input from Kinetics
            model_depth=50, # For the tutorial let's just use a 50 layer network
            model_num_class=num_classes, # Kinetics has 400 classes so we need out final head to align
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            head_pool_kernel_size=(50,4,4),
        )
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.backborn(x)
        return x