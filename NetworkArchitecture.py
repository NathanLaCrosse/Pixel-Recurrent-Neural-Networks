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




class ConditionalRowRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=32, num_layers=3, channels=1, device=torch.device('cpu')):
        super(ConditionalRowRNN, self).__init__()
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

        self.to_red = nn.Linear(hidden_size, 258)
        self.to_green = nn.Linear(hidden_size + self.embed_size, 258)
        self.to_blue = nn.Linear(hidden_size + self.embed_size * 2, 258)

        self.device = device

    def forward(self, x, target=None, temp=None):
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

        # Now, output list is the output for each layer - now we predict conditionally across color channels
        red_logits = self.to_red(output_list[0]) # Dimension: (batch_size, rows, cols, 258)
        if target is not None:
            red_embed = self.embeddings[0](target[:, 0, :, :]) # Teacher forcing
        else:
            if temp is None:
                red_pred = torch.argmax(red_logits, dim=3)
            else:
                dist = red_logits / temp
                dist = F.softmax(dist.view(-1, 258), dim=1)
                sample = torch.multinomial(dist, num_samples=1)
                red_pred = sample.view(batch_size, rows, cols)
            red_embed = self.embeddings[0](red_pred)

        # Generate green based off of results from red
        green_logits = self.to_green(torch.cat([output_list[1], red_embed], dim=3))
        if target is not None:
            green_embed = self.embeddings[1](target[:, 1, :, :])
        else:
            if temp is None:
                green_pred = torch.argmax(green_logits, dim=3)
            else:
                dist = green_logits / temp
                dist = F.softmax(dist.view(-1, 258), dim=1)
                sample = torch.multinomial(dist, num_samples=1)
                green_pred = sample.view(batch_size, rows, cols)
            green_embed = self.embeddings[1](green_pred)

        # Finally, determine blue channel based off of both previous channels
        blue_logits = self.to_blue(torch.cat([output_list[2], red_embed, green_embed], dim=3))

        if temp is not None:
            dist = blue_logits / temp
            dist = F.softmax(dist.view(-1, 258), dim=1)
            sample = torch.multinomial(dist, num_samples=1)
            blue_pred = sample.view(batch_size, rows, cols)

            return torch.stack([red_pred, green_pred, blue_pred], dim=1)

        return torch.stack([red_logits, green_logits, blue_logits], dim=1)






class GenerativeRowRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=32, num_layers=1, channels=3, device=torch.device('cpu')):
        super(GenerativeRowRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channels = channels
        self.num_layers = num_layers

        # Pixel Intensity -> Embedding Vector
        self.embeddings = nn.ModuleList(
            [nn.Embedding(257, embed_size) for i in range(channels)]
        )
        # Separate GRU for each channel, conditioned on each other
        self.channel_grus = nn.ModuleList(
            [nn.GRU(input_size=embed_size+2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) for i in
             range(channels)]
        )

        # # Embedding a previous hidden in a given color channel to something the current step can combine with the input
        # self.hidden_to_embeds = nn.ModuleList(
        #     [nn.Linear(self.hidden_size + self.embed_size + 2, self.embed_size + 2) for i in range(channels)]
        # )
        self.h_comb = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, self.hidden_size) for i in range(channels)
        ])
        self.x_comb = nn.ModuleList([
            nn.Linear(self.embed_size * 2, self.embed_size) for i in range(channels)
        ])

        self.to_red = nn.Linear(hidden_size, 257)
        self.to_green = nn.Linear(hidden_size + self.embed_size, 257)
        self.to_blue = nn.Linear(hidden_size + self.embed_size * 2, 257)

        self.device = device

    def forward(self, x):
        batch_size, channels, rows, cols = x.size()

        logits = [None for row in range(rows-1)]
        prev_hidden_row = [[torch.zeros((batch_size, self.num_layers, self.hidden_size), device=self.device) for col in range(cols - 1)] for c in range(channels)]

        for row in range(rows-1):
            current_hidden_row = [[None for col in range(cols-1)] for c in range(channels)]
            prev_hidden_left = torch.zeros((batch_size, self.num_layers, self.hidden_size), device=self.device)

            column_logits = [None for p in range(cols-1)]

            for col in range(cols-1):
                prev_color_embed = []

                comb_color_logits = [None for l in range(channels)]

                for c in range(channels):
                    # Gather previous x data from above and below
                    left_x = self.embeddings[c](x[:, c, row+1, col])
                    above_x = self.embeddings[c](x[:, c, row, col+1])
                    current_x = torch.cat([left_x, above_x], dim=1)
                    current_x = self.x_comb[c](current_x)

                    # Inject positional data
                    pos_dat = torch.tensor([row, col]).repeat(repeats=batch_size).view(batch_size, 2)
                    current_x = torch.cat([current_x, pos_dat], dim=1)

                    # Gather previous hidden data from above and below
                    current_h = torch.cat([prev_hidden_left, prev_hidden_row[c][col]], dim=2)
                    current_h = self.h_comb[c](current_h)

                    # Compute GRU output
                    _, h_final = self.channel_grus[c](current_x.view(batch_size, 1, self.embed_size),
                                                      current_h.view(self.num_layers, batch_size, self.hidden_size))
                    h_final = h_final.permute(1, 0, 2)

                    # Next up - logits conversion - based on color channel
                    colored_logits = None
                    if c == 0:
                        colored_logits = self.to_red(h_final[:, -1, :])
                    elif c == 1:
                        colored_logits = self.to_green(torch.cat([h_final[:, -1, :]] + prev_color_embed, dim=1))
                    else:
                        colored_logits = self.to_blue(torch.cat([h_final[:, -1, :]] + prev_color_embed, dim=1))

                    # Put our previous embedding in the queue for the next color channel
                    # This makes colors conditional on each other
                    prev_color_embed.append(self.embeddings[c](x[:, c, row+1, col+1]))
                    prev_hidden_left = h_final
                    current_hidden_row[c][col] = h_final

                    # Put logits in storage
                    comb_color_logits[c] = colored_logits

                column_logits[col] = torch.stack(comb_color_logits, dim=1)

            logits[row] = torch.stack(column_logits, dim=2)
            prev_hidden_row = current_hidden_row

        return torch.stack(logits, dim=2)



        # Combine logits

    def predict(self, x, mask, temp=0.01):
        # Predict pixels of an image whenever mask is false
        # Assumes x has already been padded with start-of-sequence values
        with torch.no_grad():
            batch_size, channels, rows, cols = x.size()
            im = torch.zeros((batch_size, channels, rows, cols), requires_grad=False, dtype=torch.long)

            # Append start-of sequence
            # im[:, :, 1:, 1:] = x
            im[:, :, 0, :] = 256
            im[:, :, :, 0] = 256

            prev_hidden_row = [[torch.zeros((batch_size, self.num_layers, self.hidden_size),
                                            device=self.device) for col in range(cols - 1)]
                                            for c in range(channels)]

            for row in range(rows-1):
                current_hidden_row = [[None for col in range(cols - 1)] for c in range(channels)]
                prev_hidden_left = torch.zeros((batch_size, self.num_layers, self.hidden_size), device=self.device)

                for col in range(cols-1):
                    prev_color_embed = []

                    for c in range(channels):
                        # Gather available image data
                        if not mask[:, c, row+1, col]:
                            left_x = x[:, c, row+1, col]
                        else:
                            left_x = im[:, c, row+1, col]
                        if not mask[:, c, row, col+1]:
                            above_x = x[:, c, row, col+1]
                        else:
                            above_x = im[:, c, row, col+1]

                        left_x = self.embeddings[c](left_x)
                        above_x = self.embeddings[c](above_x)
                        current_x = torch.cat([left_x, above_x], dim=1)
                        current_x = self.x_comb[c](current_x)

                        # Inject positional data
                        pos_dat = torch.tensor([row, col]).repeat(repeats=batch_size).view(batch_size, 2)
                        current_x = torch.cat([current_x, pos_dat], dim=1)

                        # Gather previous hidden data from above and below
                        current_h = torch.cat([prev_hidden_left, prev_hidden_row[c][col]], dim=2)
                        current_h = self.h_comb[c](current_h)

                        # Compute GRU output
                        _, h_final = self.channel_grus[c](current_x.view(batch_size, 1, self.embed_size),
                                                          current_h.view(self.num_layers, batch_size, self.hidden_size))
                        h_final = h_final.permute(1, 0, 2)

                        # Predict values for the current color channel - if we have to do infilling
                        if not mask[:, c, row+1, col+1]:
                            im[:, c, row+1, col+1] = x[:, c, row+1, col+1]
                        else:
                            if c == 0:
                                colored_logits = self.to_red(h_final[:, -1, :])
                            elif c == 1:
                                colored_logits = self.to_green(torch.cat([h_final[:, -1, :]] + prev_color_embed, dim=1))
                            else:
                                colored_logits = self.to_blue(torch.cat([h_final[:, -1, :]] + prev_color_embed, dim=1))

                            # Apply softmax and sample
                            colored_logits = colored_logits / temp
                            dist = F.softmax(colored_logits, dim=1)
                            pred = torch.multinomial(dist, num_samples=1)

                            im[:, c, row+1, col+1] = pred

                        # Put our previous embedding in the queue for the next color channel
                        # This makes colors conditional on each other
                        prev_color_embed.append(self.embeddings[c](im[:, c, row+1, col+1]))
                        prev_hidden_left = h_final
                        current_hidden_row[c][col] = h_final

                    pass

                prev_hidden_row = current_hidden_row


            return im





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
    net = GenerativeRowRNN(channels=3, hidden_size=32, num_layers=5)

    sample_input = torch.randint(0, 257, (5, 3, 14, 14))

    logits = net(sample_input)

    print("hi")