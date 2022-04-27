import numpy as np
import torch
import torchvision
import pandas as pd
import torchvision.transforms as tt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
# from attention import ProjectorBlock, SpatialAttn

import torch

"""
Attention blocks
Reference: Learn To Pay Attention
"""
class ProjectorBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = torch.nn.Conv2d(in_channels=in_features, out_channels=out_features,
            kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.op(x)


class SpatialAttn(torch.nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttn, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = torch.nn.Conv2d(in_channels=in_features, out_channels=1,
            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, H, W = l.size()
        c = self.op(l+g) # (batch_size,1,H,W)
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # (batch_size,C)
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,H,W), g






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

        self.maxpool1 = torch.nn.MaxPool2d(2)

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

        self.maxpool2 = torch.nn.MaxPool2d(2)

        self.cnn6 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.norm6 = torch.nn.BatchNorm2d(512)
        self.relu6 = torch.nn.ReLU(inplace=True)

        self.maxpool3 = torch.nn.MaxPool2d(2)

        # ResNet Layer 2
        self.cnn7 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm7 = torch.nn.BatchNorm2d(512)
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.cnn8 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.norm8 = torch.nn.BatchNorm2d(512)
        self.relu8 = torch.nn.ReLU(inplace=True) 

        self.maxpool4 = torch.nn.MaxPool2d(4)
        self.flatten = torch.nn.Flatten()
        self.output = torch.nn.Linear(512, 7)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        out = self.cnn3(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.norm4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.norm5(out)
        out = self.relu5(out) 
        out = self.maxpool2(out)
        out = self.cnn6(out)
        out = self.norm6(out)
        out = self.relu6(out)
        out = self.maxpool3(out)
        out = self.cnn7(out)
        out = self.norm7(out)
        out = self.relu7(out)
        out = self.cnn8(out)
        out = self.norm8(out)
        out = self.relu8(out)
        out = self.maxpool4(out)
        out = self.flatten(out)
        out = self.output(out)
        return out

class VGG16_Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # conv blocks
        self.conv1 = self.conv_block(3, 64, 2)
        self.conv2 = self.conv_block(64, 128, 2)
        self.conv3 = self.conv_block(128, 256, 3)
        self.conv4 = self.conv_block(256, 512, 3)
        self.conv5 = self.conv_block(512, 512, 3)
        self.conv6 = self.conv_block(512, 512, 2, pool=True)
        self.dense = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True)
        # attention blocks
        self.projector = ProjectorBlock(256, 512)
        self.attn1 = SpatialAttn(in_features=512, normalize_attn=True)
        self.attn2 = SpatialAttn(in_features=512, normalize_attn=True)
        self.attn3 = SpatialAttn(in_features=512, normalize_attn=True)
        # final classification layer

        self.classify = torch.nn.Linear(in_features=512*3, out_features=7, bias=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        l1 = self.conv3(x)
        x = F.max_pool2d(l1, kernel_size=2, stride=2, padding=0)
        l2 = self.conv4(x)
        x = F.max_pool2d(l2, kernel_size=2, stride=2, padding=0)
        l3 = self.conv5(x)
        x = F.max_pool2d(l3, kernel_size=2, stride=2, padding=0)
        x = self.conv6(x)
        g = self.dense(x) # batch_sizex512x1x1
        # attention
        c1, g1 = self.attn1(self.projector(l1), g)
        c2, g2 = self.attn2(l2, g)
        c3, g3 = self.attn3(l3, g)
        g = torch.cat((g1,g2,g3), dim=1) # batch_sizex3C
        # classification layer
        x = self.classify(g) # batch_sizexnum_classes
        return x

    def conv_block(self, in_features, out_features, blocks, pool=False):
        layers = []
        for i in range(blocks):
            conv2d = torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, torch.nn.BatchNorm2d(out_features), torch.nn.ReLU(inplace=True)]
            in_features = out_features
            if pool:
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        return torch.nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Sentiment Analysis with age and gender')
    # parser.add_argument('--epochs', default=300, type=int)
    # parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--lr', default=0.1, type=float)

    # Loading data from data folder /data/sentiment
#     DATA_DIR = "data/sentiment/"
    DATA_DIR = "../input/facial-expression-dataset-image-folders-fer2013/data/"

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
                            tt.ToTensor()])
    valid_transform = tt.Compose([tt.ToTensor()])

    # Create train and valid datasets
    train = torchvision.datasets.ImageFolder(TRAIN_DIR, train_transform)
    valid = torchvision.datasets.ImageFolder(VAL_DIR, valid_transform)

    batch_size = 128
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

    # model = BaseLine()
    # model = ResNet()
    model = VGG16_Attn()
    model.cuda()
    torch.cuda.empty_cache()
    for i in range(epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        score = 0
        model.train()
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
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
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()    
            out = model(images)
            val_loss = loss_function(out, labels)
            _, pred_val = torch.max(out,axis=1)
            score_val += (pred_val == labels).sum()
        val_acc = score_val / len(valid)
        
        train_loss.append(loss)
        #val_loss.append(val_loss)
        train_acc.append(acc)
        print("{}/{} Epochs  | Train Loss={:.4f}  |Train_Accuracy={:.4f} |Val_loss={:.4f}  |Val_Accuracy={:.4f}".format(i+1,epoch,loss,acc,val_loss,val_acc)  ) 
    torch.save(model, "VGG16_Attention_bs128_lr0.001.pt")