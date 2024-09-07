# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import os
import random

import numpy as np
import torch

import pickle

def DumpPick(data_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)

# 从文件中读取
def LoadPick(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


def setup_seed(seed=2022):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device
