import torch
from .CrashModel import *
from .Resnet3D import *
from .ConvEmbeding import *
from .WTModel import *

def create_model(model_name:str, number_of_class:int)-> torch.nn.Module:
    if model_name == "Resnet3D_v1":
        return Resnet3D_v1(number_of_class)
    if model_name == "Resnet3D_v2":
        return Resnet3D_v2(number_of_class)
    elif model_name == "ConvEmbeding_v1":
        return ConvEmbeding_v1(number_of_class)
    elif model_name == "ConvEmbeding_v2":
        return ConvEmbeding_v2(number_of_class)
    elif model_name == "ConvEmbeding_v3":
        return ConvEmbeding_v3(number_of_class)
    elif model_name == "WTModel_v1":
        return WTModel_v1(number_of_class)
    else:
        raise Exception("Unknown model name!")