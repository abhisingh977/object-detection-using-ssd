import time
from typing import List, Any
import wandb
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import argparse
# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects

torch.cuda.set_device(0)
torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import io

import numpy as np

a = np.array([3, 4, 5])
stream = io.BytesIO(a.tobytes())  # implements seek()
torch.load(stream)

PATH='C:/Users/Home/Documents/res/res/model.pt'
# Learning
checkpoint = torch.load(PATH)# path to model checkpoint, None if none
#stream = io.BytesIO(checkpoint.tobytes())
batch_size = 16 # batch size