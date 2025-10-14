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
    def __init__(self, embed_size=64, hidden_size=32, num_layers=3, channels=1, device=torch.device('cpu')):
        super(RowRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channels = channels

        # Pixel Intensity -> Embedding Vector
        self.embeddings = nn.ModuleList(
            [nn.Embedding(258, embed_size) for i in range(channels)]
        )
        # Separate GRU for each channel, conditioned on each other
        self.channel_grus = nn.ModuleList(
            [nn.GRU(input_size=embed_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) for i in range(channels)]
        )
        # Embedding a previous hidden in a given color channel to something the current step can combine with the input
        self.hidden_to_embeds = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.embed_size+1) for i in range(channels)]
        )
        # Way of conditioning channels -> translate information from one channel to the next
        self.channel_adapters = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.embed_size+1) for i in range(channels - 1)]
        )
        self.to_out = nn.Linear(hidden_size, 258)
        self.device = device

    def forward(self, x):
        batch_size, channels, rows, cols = x.size()

        # Embed and add row signature to the data
        expanded_x = torch.zeros((batch_size, channels, rows, cols, self.embed_size+1), device=self.device)
        for c in range(channels):
            embedded = self.embeddings[c](x[:, c, :, :])
            row_data = torch.arange(0, rows, device=self.device) / rows * 2 - 1
            row_data = row_data.repeat(batch_size, 1, cols, 1).permute(0, 3, 2, 1)

            expanded_x[:, c, :, :, :] = torch.cat((embedded, row_data), dim=3)
        x = expanded_x

        # Save output to memory for later reference
        output_list = [[None for _ in range(rows)] for c in range(channels)]

        # For each color channel, we run one of the grus, adding in previous color channel info when needed
        for c in range(channels):
            # Initialize previous hidden vectors
            prev_hiddens = torch.zeros((batch_size, cols, self.hidden_size), device=self.device)

            # For each row, calculate hiddens. Add previous hidden row data.
            for row in range(rows):
                cur_row = x[:, c, row, :, :]
                comb = cur_row + self.hidden_to_embeds[c](prev_hiddens)

                # For later color channels, add data from the previous hidden at this spot
                if c > 0:
                    # comb = comb + self.channel_adapters[c-1](outputs[:, c-1, row, :, :])
                    comb = comb + self.channel_adapters[c - 1](output_list[c-1][row])

                # Calculate GRU output
                gru_output, _ = self.channel_grus[c](comb)
                prev_hiddens = gru_output
                # outputs[:, c, row, :, :] = gru_output
                output_list[c][row] = gru_output

        # Stack up to proper size
        for c in range(channels):
            output_list[c] = torch.stack(output_list[c], dim=1)  # rows
        outputs = torch.stack(output_list, dim=1)

        # Apply conversion to pixel intensities
        pred = self.to_out(outputs)
        return pred


if __name__ == "__main__":
    net = RowRNN(channels=3)

    sample_input = torch.randint(0, 258, (5, 1, 14, 14))

    logits = net(sample_input)

    print("hi")