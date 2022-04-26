import numpy as np
import torch
import torchvision
import os
import pandas as pd
import torchvision.transforms as tt
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from utils import get_args, check_args

class BaseLine(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.relu1 = torch.nn.ReLU()
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=3)
        self.cnn2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=3)
        self.cnn3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.relu3 = torch.nn.ReLU()
        self.avgpool3 = torch.nn.AvgPool2d(kernel_size=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.relu4 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 256)
        self.relu5 = torch.nn.ReLU()
        self.output = torch.nn.Linear(256, 7)
    
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.avgpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.avgpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.avgpool3(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu4(out)
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        out = self.relu5(out)
        out = self.output(out)
        return out 

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=True)
        
        self.cnn2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.avgpool1 = torch.nn.AvgPool2d(2)

        # ResNet Layer 1
        self.cnn3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.cnn4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm4 = torch.nn.BatchNorm2d(128)
        self.relu4 = torch.nn.ReLU(inplace=True)

        self.cnn5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.norm5 = torch.nn.BatchNorm2d(256)
        self.relu5 = torch.nn.ReLU(inplace=True) 

        self.avgpool2 = torch.nn.AvgPool2d(2)

        self.cnn6 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.norm6 = torch.nn.BatchNorm2d(512)
        self.relu6 = torch.nn.ReLU(inplace=True)

        self.avgpool3 = torch.nn.AvgPool2d(2)

        # ResNet Layer 2
        self.cnn7 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm7 = torch.nn.BatchNorm2d(512)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.cnn8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm8 = torch.nn.BatchNorm2d(512)
        self.relu8 = torch.nn.ReLU(inplace=True) 

        self.avgpool4 = torch.nn.AvgPool2d(4)
        self.flatten = torch.nn.Flatten()
        self.output = torch.nn.Linear(512, 7)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.avgpool1(out)
        out = self.cnn3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.norm4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.norm5(out)
        out = self.relu5(out) 
        out = self.avgpool2(out)
        out = self.cnn6(out)
        out = self.norm6(out)
        out = self.relu6(out)
        out = self.avgpool3(out)
        out = self.cnn7(out)
        out = self.norm7(out)
        out = self.relu7(out)
        out = self.cnn8(out)
        out = self.norm8(out)
        out = self.relu8(out)
        out = self.avgpool4(out)
        out = self.flatten(out)
        out = self.output(out)
        return out

if __name__ == "__main__":
    # args = get_args()
    # check_args(args)

    # Loading data from data folder /data/sentiment
    DATA_DIR = "data/sentiment/"
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
    train_transform = tt.Compose([tt.RandomHorizontalFlip(), tt.RandomRotation(10),
                            tt.ToTensor(), tt.Normalize((0.5,),(0.5,))])
    valid_transform = tt.Compose([tt.ToTensor()])

    # Create train and valid datasets
    train = torchvision.datasets.ImageFolder(TRAIN_DIR, train_transform)
    valid = torchvision.datasets.ImageFolder(VAL_DIR, valid_transform)

    batch_size = 512
    epoch = 10
    learning_rate = 0.001

    # Load data for training and validation
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size * 2, num_workers=3, pin_memory=True)


    loss_function = F.cross_entropy
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]

    # base_model = BaseLine()
    # for i in range(epoch):
    #     optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
    #     score = 0
    #     base_model.train()
    #     for images, labels in train_loader:
    #         out = base_model(images)
    #         loss = loss_function(out, labels)
    #         base_model.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         _, pred = torch.max(out,axis=1)
    #         score += (pred==labels).sum()
    #     acc=score/len(train)

    #     score_val=0
    #     for images,labels in valid_loader:
    #         out = base_model(images)
    #         val_loss = loss_function(out, labels)
    #         _, pred_val = torch.max(out,axis=1)
    #         score_val += (pred_val == labels).sum()
    #     val_acc = score_val / len(valid)
        
    #     train_loss.append(loss)
    #     #val_loss.append(val_loss)
    #     train_acc.append(acc)
    #     print("{}/{} Epochs  | Train Loss={:.4f}  |Train_Accuracy={:.4f} |Val_loss={:.4f}  |Val_Accuracy={:.4f}".format(i+1,epoch,loss,acc,val_loss,val_acc)  ) 

    # torch.save(base_model, MODEL_DIR+"base_model_bs512_lr0.001.pt")

    model = ResNet()
    for i in range(epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        score = 0
        model.train()
        for images, labels in train_loader:
            out = model(images)
            loss = loss_function(out, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out,axis=1)
            score += (pred==labels).sum()
        acc=score/len(train)

        score_val=0
        for images,labels in valid_loader:
            out = model(images)
            val_loss = loss_function(out, labels)
            _, pred_val = torch.max(out,axis=1)
            score_val += (pred_val == labels).sum()
        val_acc = score_val / len(valid)
        
        train_loss.append(loss)
        #val_loss.append(val_loss)
        train_acc.append(acc)
        print("{}/{} Epochs  | Train Loss={:.4f}  |Train_Accuracy={:.4f} |Val_loss={:.4f}  |Val_Accuracy={:.4f}".format(i+1,epoch,loss,acc,val_loss,val_acc)  ) 
