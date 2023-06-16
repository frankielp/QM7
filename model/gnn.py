import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch
import os
from torch.autograd import Variable
from scipy.io import loadmat
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self,batch,device):
        super(GCN, self).__init__()
        self.batch=batch
        self.W1 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).type(torch.FloatTensor).to(device), requires_grad=True)
        # self.W2 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).type(torch.FloatTensor).to(device), requires_grad=True)
        # self.W3 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).type(torch.FloatTensor).to(device), requires_grad=True)
        # self.W4 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).type(torch.FloatTensor).to(device), requires_grad=True)
        # self.W5 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).to(device), requires_grad=True)
        # self.W6 = torch.nn.Parameter(torch.randn(self.batch, 23, 23).to(device), requires_grad=True)
        self.W7 = torch.nn.Parameter(torch.randn(self.batch, 23, 5).type(torch.FloatTensor).to(device), requires_grad=True)
        self.W8 = torch.nn.Parameter(torch.randn(23*5, 1).type(torch.FloatTensor).to(device), requires_grad=True)

    def forward(self, x, A, D):
        # Symmetric Normalization
        hidden_layer_1 = F.leaky_relu(D.bmm(A).bmm(D).bmm(x).bmm(self.W1))
        # hidden_layer_2 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_1).bmm(self.W2))
        # hidden_layer_3 = F.sigmoid(D.bmm(A).bmm(D).bmm(hidden_layer_2).bmm(self.W3))
        # hidden_layer_4 = F.sigmoid(D.bmm(A).bmm(D).bmm(hidden_layer_3).bmm(self.W4))
        # hidden_layer_5 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_4).bmm(self.W5))
        # hidden_layer_6 = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_5).bmm(self.W6))
        y_pred = F.leaky_relu(D.bmm(A).bmm(D).bmm(hidden_layer_1).bmm(self.W7))
        y_pred = y_pred.view(self.batch, 23*5)
        y_pred = F.leaky_relu(y_pred.mm(self.W8))
        return y_pred


if __name__=="__main__":
    raise NotImplementedError