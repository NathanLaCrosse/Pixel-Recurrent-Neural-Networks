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







class OmniRowRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=32, num_layers=3, channels=1, device=torch.device('cpu')):
        super(OmniRowRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channels = channels

        self.direction_ref = ["Top-Left","Top-Right","Bottom-Left","Bottom-Right"]

        self.direction_path = {
            "Top-Left" : (1, 1), # Row increments ascendingly, so does col
            "Top-Right" : (1, -1),
            "Bottom-Left" : (-1, 1),
            "Bottom-Right" : (-1, -1)
        }

        # Pixel Intensity -> Embedding Vector
        self.embeddings = nn.ModuleDict({ref : nn.ModuleList(
            [nn.Embedding(258, embed_size) for i in range(channels)]
        ) for ref in self.direction_ref})
        # Separate GRU for each channel, conditioned on each other
        self.channel_grus = nn.ModuleDict({ref : nn.ModuleList(
            [nn.GRU(input_size=embed_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) for i in range(channels)]
        ) for ref in self.direction_ref})
        # Embedding a previous hidden in a given color channel to something the current step can combine with the input
        self.hidden_to_embeds = nn.ModuleDict({ref : nn.ModuleList(
            [nn.Linear(self.hidden_size, self.embed_size+1) for i in range(channels)]
        ) for ref in self.direction_ref})
        # Way of conditioning channels -> translate information from one channel to the next
        self.channel_adapters = nn.ModuleDict({ref : nn.ModuleList(
            [nn.Linear(self.hidden_size, self.embed_size+1) for i in range(channels - 1)]
        ) for ref in self.direction_ref})
        self.to_out = nn.Linear(4*hidden_size, 258)
        self.device = device

    def forward(self, x):
        batch_size, channels, rows, cols = x.size()

        # Embed and add row signature to the data
        expanded_x = torch.zeros((batch_size, 4, channels, rows, cols, self.embed_size+1), device=self.device)
        for d in range(len(self.direction_ref)):
            ref = self.direction_ref[d]
            for c in range(channels):
                embedded = self.embeddings[ref][c](x[:, c, :, :])
                row_data = torch.arange(0, rows, device=self.device) / rows * 2 - 1
                row_data = row_data.repeat(batch_size, 1, cols, 1).permute(0, 3, 2, 1)

                expanded_x[:, d, c, :, :, :] = torch.cat((embedded, row_data), dim=3)
        x = expanded_x

        # Save output to memory for later reference
        output_list = [[[None for _ in range(rows)] for c in range(channels)] for d in range(len(self.direction_ref))]

        # For each color channel, we run one of the grus, adding in previous color channel info when needed
        for d in range(len(self.direction_ref)):
            ref = self.direction_ref[d]
            for c in range(channels):
                # Initialize previous hidden vectors
                prev_hiddens = torch.zeros((batch_size, cols, self.hidden_size), device=self.device)

                # For each row, calculate hiddens. Add previous hidden row data.
                for row in range(rows):
                    grabbed_row = row if self.direction_path[ref][0] == 1 else rows - 1 - row
                    reverse = self.direction_path[ref][1] == -1

                    cur_row = x[:, d, c, grabbed_row, :, :]
                    if reverse:
                        cur_row = cur_row.flip(dims=[1])
                    comb = cur_row + self.hidden_to_embeds[ref][c](prev_hiddens)

                    # For later color channels, add data from the previous hidden at this spot
                    if c > 0:
                        past_data = output_list[d][c-1][grabbed_row][:, :, :]
                        if reverse:
                            past_data = past_data.flip(dims=[1])
                        comb = comb + self.channel_adapters[ref][c-1](past_data)

                    # Calculate GRU output
                    gru_output, _ = self.channel_grus[ref][c](comb)
                    prev_hiddens = gru_output

                    if reverse:
                        gru_output = gru_output.flip(dims=[1])

                    output_list[d][c][grabbed_row] = gru_output

        # Stack up to proper size
        for d in range(len(self.direction_ref)):
            for c in range(channels):
                output_list[d][c] = torch.stack(output_list[d][c], dim = 1)
        for d in range(len(self.direction_ref)):
            output_list[d] = torch.stack(output_list[d], dim=1)  # rows
        outputs = torch.stack(output_list, dim=1)

        outputs = outputs.view(batch_size, channels, rows, cols, self.hidden_size*4)

        # Apply conversion to pixel intensities
        pred = self.to_out(outputs)
        return pred







if __name__ == "__main__":
    net = OmniRowRNN(channels=3, hidden_size=32)

    sample_input = torch.randint(0, 258, (5, 3, 14, 14))

    logits = net(sample_input)

    print("hi")