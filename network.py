# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Kim-Louis Simmoteit, Oliver Drozdowski
"""

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from dataloader import *

