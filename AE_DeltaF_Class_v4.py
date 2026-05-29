import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

import Utilities  # Custom module for loading settings, data, and handling model folders

# Define the Autoencoder model with variable hidden depth

# input to ae and enc is size (B, MM**3) - 2d
# output of ene and input of dec is size (B, code_len) - 2d
# ouput of dec and ae is size (B, MM**3) - 2d

# Define the Autoencoder model with variable hidden depth
class Autoencoder(nn.Module):
    def __init__(self, solsize, code_len, hidden_layer_num, dropout=0.0, bias=False):
        super(Autoencoder, self).__init__()
        self.dropout = dropout

        if hidden_layer_num == 1:
            self.encoder = nn.Sequential(
                nn.Linear(solsize, code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            self.decoder = nn.Sequential(
                nn.Linear(code_len, solsize, bias=bias),
                nn.Dropout(dropout)
            )

        elif hidden_layer_num == 3:
            self.encoder = nn.Sequential(
                nn.Linear(solsize, 2*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*code_len, code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            self.decoder = nn.Sequential(
                nn.Linear(code_len, 2*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*code_len, solsize, bias=bias),
            )

        else:
            self.encoder = nn.Sequential(
                nn.Linear(solsize, 4*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(4*code_len, 2*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*code_len, code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            )
            self.decoder = nn.Sequential(
                nn.Linear(code_len, 2*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*code_len, 4*code_len, bias=bias),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(4*code_len, solsize, bias=bias),
            )

    def forward(self, x):
        return self.decoder(self.encoder(x))
