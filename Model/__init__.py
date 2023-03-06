import torch
from .CrashModel import *
from .Resnet3D import *

def create_model(model_name:str, number_of_class:int)-> torch.nn.Module:
    if model_name == "Resnet3D_v1":
        return Resnet3D_v1(number_of_class)
    else:
        raise Exception("Unknown model name!")