#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleConv3Net(torch.nn.Module):
    def __init__(self):
        super(SimpleConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 27 * 27, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("X shape: ", x.shape)
        x = x.view(-1, 256 * 27 * 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(self.fc2(x))

        return x


if __name__ == "__main__":
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 3, 224, 224))
    model = SimpleConv3Net()
    y = model(x)
    print(y)
