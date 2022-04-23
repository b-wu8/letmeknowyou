from ast import arg
import numpy as np
import torch
import torchvision
import os
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from utils import get_args, check_args

# args = get_args()
# check_args(args)

# Loading data from data folder /data/sentiment
DATA_DIR = "data/sentiment/"

TRAIN_DIR = DATA_DIR + "train/"
VAL_DIR = DATA_DIR + "val/"

# There is a total of 6 classes of facial expressions, labeled from 0 to 6
classes = os.listdir(TRAIN_DIR)
print(classes)

# Classes of facial expressions
# Anger = 0
# Digust = 1
# Fear = 2
# Happy = 3
# Sad = 4
# Surprise = 5
# Neutal = 6
expression = ['Anger','Disgust','Fear','Happy','Sad','Suprise','Neutral'] 

# Transform data, normalization
train_transform = tt.Compose([tt.RandomHorizontalFlip(), tt.RandomRotation(10),
                         tt.ToTensor()])
valid_transform = tt.Compose([tt.ToTensor()])

# Create train and valid datasets
train = torchvision.datasets.ImageFolder(TRAIN_DIR, train_transform)
valid = torchvision.datasets.ImageFolder(VAL_DIR, valid_transform)

batch_size = 128

# Load data for training and validation
train = DataLoader(train, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid = DataLoader(valid, batch_size * 2, num_workers=3, pin_memory=True)

# Algorithm structure
###