import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """
    reference : https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma:float=2.0, alpha:float=0.25, size_average:bool=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        # self.alpha = torch.Tensor([alpha,1-alpha])
        self.size_average = size_average

    def forward(self, input, target):
        """
        Focal Loss : -at * (1 - pt)^gamma * log(pt) 
        """
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        print(f"logpt : {logpt.shape} - {logpt} / target : {target}")
        logpt = logpt.gather(dim=0, index=target)
        logpt = logpt.view(-1)

        pt = Variable(logpt.data.exp())
        logpt = Variable(logpt)

        # at = self.alpha.gather(0,target.data.view(-1))
        # at = Variable(self.alpha)

        loss = -1 * self.alpha * ((1-pt)**self.gamma) * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()