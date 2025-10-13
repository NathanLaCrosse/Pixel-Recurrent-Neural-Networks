import os
import torch
from torch import nn
import pandas as pd
import tqdm as tqdm
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader
import MNISTData as md
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

class RowRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=32, num_layers=3, device=torch.device('cpu')):
        super(RowRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(258, embed_size)
        self.gru = nn.GRU(input_size=embed_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_to_embed = nn.Linear(self.hidden_size, self.embed_size+1)
        self.to_out = nn.Linear(hidden_size, 258)
        self.device = device

    def forward(self, x):
        batch_size, channels, rows, cols = x.size()

        # Add row signature to the row
        x = x.view(batch_size, rows, cols)
        x = self.embedding(x) # Now size is batch_size x rows x cols x embed_size
        row_data = torch.arange(0, rows, device=self.device) / rows * 2 - 1
        row_data = row_data.repeat(batch_size, 1, cols, 1).permute(0, 3, 2, 1)
        x = torch.cat((x, row_data), dim=3)

        # Current x size: (batch_size x rows x cols x embed_size+1)
        outputs = torch.zeros((batch_size, rows, cols, self.hidden_size), device=self.device)
        prev_hiddens = torch.zeros((batch_size, cols, self.hidden_size), device=self.device)

        # For each row, calculate hiddens then add those hiddens to the next row
        for row in range(rows):
            # Combine hidden on previous row with x
            cur_row = x[:, row, :, :]
            comb = cur_row + self.hidden_to_embed(prev_hiddens)

            # Calculate GRU output
            gru_output, _ = self.gru(comb)
            prev_hiddens = gru_output
            outputs[:, row, :, :] = gru_output

        # Apply conversion to pixel intensities
        pred = self.to_out(outputs)
        return pred.view(batch_size, channels, rows, cols, 258)