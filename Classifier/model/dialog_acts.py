import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class Encoder(nn.Module):
    def __init__(self, D, classes_num):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D, 1024)
        self.linear2 = nn.Linear(1024, 300)
        self.linear3 = nn.Linear(300, classes_num)

    # @torchsnooper.snoop()
    def forward(self, x):

        x = F.dropout(x, p=0.5)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        classes = T.sigmoid(self.linear3(x))

        return classes
