import numpy as np
import os
import Utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
import matplotlib.pyplot as plt

class LearnColOpCNN(nn.Module):
    def __init__(self, dms, output_dim, bias, dropout_rate=0.0):
        super(LearnColOpCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=5, padding=2, bias=bias),
            nn.PReLU(),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(4, 8, kernel_size=3, padding=1, bias=bias),
            nn.PReLU(),
            nn.Dropout3d(dropout_rate),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=bias),
            nn.PReLU(),
            nn.Dropout3d(dropout_rate),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=bias),
            nn.PReLU(),
            nn.Dropout3d(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear((dms//4)**3 * 32, output_dim, bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
