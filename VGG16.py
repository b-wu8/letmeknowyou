##########################
### MODEL VGG16 Kunliang 04/26
##########################

import numpy as np
import torch
import torchvision
import os
import pandas as pd
import torchvision.transforms as tt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import get_args, check_args
from tqdm import tqdm

class VGG16(torch.nn.Module):

    def __init__(self, num_classes, height, width):
        super().__init__()
        self.height = height
        self.width = width

        self.block_1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2), # in max-pooling we reduce the size
                                   stride=(2, 2))
        )

        self.block_2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2))
        )


        self.block_4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2))
        )

        self.block_5 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2),
                                   stride=(2, 2))
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * self.height * self.width, 4096),
            torch.nn.ReLU(True),
            # torch.nn.Dropout(p=0.5), # 1st
            torch.nn.Dropout2d(p=0.2),
            # torch.nn.Linear(4096, 1024), # 1st
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            # torch.nn.Dropout2d(p=0.5), # 1st
            torch.nn.Linear(4096, num_classes),
        )

        for m in self.modules(): # what's included in the paper
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.avgpool(x)
        x = x.view(-1, 512 * self.height * self.width) # flatten

        out = self.classifier(x)

        return out


if __name__ == "__main__":
    # args = get_args()
    # check_args(args)

    # Loading data from data folder /data/sentiment
    DATA_DIR = "data/"
    # LABEL_DIR = "data/sentiment/fer2013.csv"

    TRAIN_DIR = DATA_DIR + "train/"
    VAL_DIR = DATA_DIR + "val/"
    MODEL_DIR = "models/"

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
    train_transform = tt.Compose([
        tt.Resize((72, 72)), # slightly upscale the pictures since VGG16 has too many more layers for small inputs
        tt.RandomCrop((64, 64)), # random crop the images to 64 * 64 to reduce the overfitting
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize so that the pictures centered at 0. still doubtful 3 48 48

    valid_transform = tt.Compose([
        tt.Resize((72, 72)),
        tt.CenterCrop((64, 64)),
        tt.ToTensor(),
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Create train and valid datasets
    train = torchvision.datasets.ImageFolder(TRAIN_DIR, train_transform)
    valid = torchvision.datasets.ImageFolder(VAL_DIR, valid_transform)

    batch_size = 512
    epoch = 10
    learning_rate = 0.001

    # Load data for training and validation
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size * 2, num_workers=3, pin_memory=True)

    model_vgg16 = VGG16(num_classes=len(expression),height=4,width=4)

    loss_function = F.cross_entropy
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    for i in tqdm(range(epoch)):
        print('starting the {} epoch'.format(i+1))
        optimizer = torch.optim.Adam(model_vgg16.parameters(), lr=learning_rate)
        score = 0
        model_vgg16.train()
        for images, labels in train_loader: # batch size 512
            out = model_vgg16(images)
            loss = loss_function(out, labels)
            model_vgg16.zero_grad() # Sets the gradients of all optimized torch. Tensor s to zero.
            loss.backward() # backward propagation
            optimizer.step() # After computing the gradients for all tensors in the model, calling optimizer. step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
            _, pred = torch.max(out, axis=1) # might figured out
            score += (pred==labels).sum()
        acc=score/len(train)

        score_val=0
        for images,labels in valid_loader:
            out = model_vgg16(images)
            val_loss = loss_function(out, labels)
            _, pred_val = torch.max(out, axis=1)
            score_val += (pred_val == labels).sum()
        val_acc = score_val / len(valid)

        train_loss.append(loss)
        #val_loss.append(val_loss)
        train_acc.append(acc)
        print("{}/{} Epochs  | Train Loss={:.4f}  |Train_Accuracy={:.4f} |Val_loss={:.4f}  |Val_Accuracy={:.4f}".format(i+1,epoch,loss,acc,val_loss,val_acc)  )

    torch.save(model_vgg16, MODEL_DIR+"VGG16_model_bs512_lr0.001.pt")
