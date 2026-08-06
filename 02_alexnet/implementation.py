import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Single GPU Implementation
        self.conv1 = nn.Conv2d(3,96,(11,11),4,padding=2)
        self.conv2 = nn.Conv2d(96,256,(5,5),padding=2)
        self.conv3 = nn.Conv2d(256,384,(3,3),padding=1)
        self.conv4 = nn.Conv2d(384,384,(3,3),padding=1)
        self.conv5 = nn.Conv2d(384,256,(3,3),padding=1)

        self.dropout = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(9216,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,100)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

        # Initialize neuron biases to 1: Conv2, Conv4, Conv5, all fc layers
        
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

        nn.init.constant_(self.fc6.bias, 1)
        nn.init.constant_(self.fc7.bias, 1)
        nn.init.constant_(self.fc8.bias, 0)

    
    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        n1 = F.local_response_norm(c1,size=5,alpha=1e-4,beta=0.75,k=2)
        s1 = F.max_pool2d(n1,kernel_size=3,stride=2)

        c2 = F.relu(self.conv2(s1))
        n2 = F.local_response_norm(c2,size=5,alpha=1e-4,beta=0.75,k=2)
        s2 = F.max_pool2d(n2,kernel_size=3,stride=2)

        c3 = F.relu(self.conv3(s2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))

        s3 = F.max_pool2d(c5,kernel_size=3,stride=2)

        s4 = torch.flatten(s3, start_dim=1)

        fc6 = F.relu(self.fc6(s4))
        x = self.dropout(fc6)
        fc7 = F.relu(self.fc7(x))
        x = self.dropout(fc7)

        fc8 = self.fc8(x)


        return fc8



