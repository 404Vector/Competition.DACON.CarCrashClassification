import torch
from .CrashModel import *
from .Resnet3D import *
from .ConvEmbeding import *
from .WTModel import *

def create_model(model_name:str, number_of_class:int)-> torch.nn.Module:
    return eval(f'{model_name}({number_of_class})')
