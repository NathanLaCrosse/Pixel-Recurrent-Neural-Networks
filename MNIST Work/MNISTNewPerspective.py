import os
import torch
from torch import nn
import pandas as pd
import tqdm as tqdm
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader
import Data as md
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

class MNISTImages(Dataset):
    def __init__(self,filepath="../Datasets/mnist_train.csv", samples=60000):
        _, self.raw_images = md.load_mnist(filepath=filepath, samples=samples)

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, item):
        im = torch.tensor(self.raw_images[item], dtype=torch.long)
        start_of_col = torch.ones((28, 1), dtype=torch.long) * 256
        im = torch.cat((start_of_col, im), dim=1)
        return im.view(1, 28, 29)


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
        # x = x.view(batch_size*rows, cols, self.embed_size+1)

        # Current x size: (batch_size x rows x cols x embed_size+1)
        outputs = torch.zeros((batch_size, rows, cols, self.hidden_size), device=self.device)
        prev_hiddens = torch.zeros((batch_size, cols, self.hidden_size), device=self.device)

        # For each row, calculate hiddens then add those hiddens to the next row
        for row in range(rows):
            cur_row = x[:, row, :, :]
            comb = cur_row + self.hidden_to_embed(prev_hiddens)
            gru_output, _ = self.gru(comb)
            prev_hiddens = gru_output
            outputs[:, row, :, :] = gru_output

        # Gather all of the hidden vectors
        # outputs, _ = self.gru(x)

        # Apply conversion to pixel intensities
        pred = self.to_out(outputs)
        return pred.view(batch_size, channels, rows, cols, 258)

    # def sample(self, batch_size):
    #     sampled = torch.zeros((batch_size, 1, 28, 28), dtype=torch.long)
    #     start_of_col = torch.ones((batch_size, 1, 28, 1), dtype=torch.long) * 256
    #     sampled = torch.cat((start_of_col, sampled), dim=3)
    #
    #     with torch.no_grad():
    #         for row in range(28):
    #             for col in range(29):
    #                 # Grab logit for this pixel
    #                 logits = self.forward(sampled)
    #                 pixel_dist = F.softmax(logits[:, 0, row, col, :], dim=-1)
    #
    #                 # Sample from the distribution
    #                 sample = torch.multinomial(pixel_dist, num_samples=1)
    #
    #                 sampled[:, :, row, col] = sample
    #
    #     return sampled

    # def infill(self, x, missing_mask, temp=1):
    #     dest = torch.clone(x)
    #     batch_size, channels, rows, cols = x.size()
    #
    #     with torch.no_grad():
    #         for row in range(rows):
    #             for col in range(cols):
    #                 if np.any(missing_mask[:, :, row, col]):
    #                     # Grab logits for this pixel(s)
    #                     logits = self.forward(dest) / temp
    #                     pixel_dist = F.softmax(logits[:, 0, row, col, :], dim=-1)
    #
    #                     sample = torch.multinomial(pixel_dist, num_samples=1)
    #
    #                     dest[:, :, row, col] = sample
    #
    #     return dest

def train_infill_model(epochs, batch_size, embed_size, hidden_size, numlayers, save_file="InfillRNN.pt", infill_pixel_count=3, infill_increment=3, infill_grid_max=4, current_grid_max=1, max_infill_pixels=28*28*0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RowRNN(embed_size=embed_size, hidden_size=hidden_size, num_layers=numlayers, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net = net.to(device=device)

    for epoch in range(epochs):
        dat_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
        progress_bar = tqdm.tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for _, batch in enumerate(progress_bar):
            ims = batch

            # Obstruct random pixels from the image
            remaining_infill = infill_pixel_count
            obstructed = torch.clone(ims)
            while remaining_infill > 0:
                block_size = np.random.randint(1,current_grid_max+1)

                rand_row = np.random.randint(0,28)
                rand_col = np.random.randint(0,28)

                obstructed[:, :, rand_row:rand_row+block_size, rand_col+1:rand_col+1+block_size] = 257
                remaining_infill -= block_size**2

            # for it in range(infill_pixel_count):
            #     rand_row = np.random.randint(0,28)
            #     rand_col = np.random.randint(0,28)

            #     obstructed[:, :, rand_row:rand_row+2, rand_col+1:rand_col+3] = 257
            obstructed = obstructed.to(device)

            optimizer.zero_grad()
            logits = net(obstructed) # Result: batch_size, 1, 28, 29, 258

            # Clip out start-of-sequence blip
            logits = logits[:,:,:,1:,:]
            ims = ims[:,:,:,1:]

            loss = loss_fn(logits.reshape(-1, 258), ims.reshape(-1).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss:": loss.item()})

        infill_pixel_count += infill_increment
        infill_pixel_count = min(infill_pixel_count, max_infill_pixels)

        if (epoch + 1) % 10 == 0:
            current_grid_max = min(current_grid_max + 1, infill_grid_max)

        torch.save(net.state_dict(), save_file)

    return running_loss

# ---------- Training Code ----------
# dat = MNISTImages()

# epochs = 100
# batch_size = 512

# infill_pixel_count = 3
# infill_increment = 3
# infill_grid_max = 4
# current_grid_max = 1
# max_infill_pixels = 28 * 28 * 0.2

# final_loss = []

# final_loss.append(train_infill_model(epochs, batch_size, embed_size=64, hidden_size=32, numlayers=3, save_file="BasicInfill"))
# final_loss.append(train_infill_model(epochs, batch_size,embed_size=32, hidden_size=32, numlayers=3, save_file="SmallerEmbedInfill"))
# final_loss.append(train_infill_model(epochs, batch_size,embed_size=64, hidden_size=64, numlayers=5, save_file="LargerInfill"))
# final_loss.append(train_infill_model(epochs, batch_size//2,embed_size=64, hidden_size=128, numlayers=10, save_file="DeepInfill"))

# for i in range(len(final_loss)):
#     print(f"Loss ({i+1}): {final_loss[i]}")

# ---------- Testing Code ----------
net = RowRNN(embed_size=64, hidden_size=64, num_layers=5)
# net = RowRNN(embed_size=64, hidden_size=128, num_layers=10)
state_dict = torch.load("../Models/LargerInfill.pt", map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()

infill_pixel_count = 28*28//2

# Classic reconstruction. (Sanity Check)
dat = MNISTImages(filepath="../Datasets/mnist_test.csv")
with torch.no_grad():
    for im in dat:
        obstructed = im.view(1, 1, 28, 29)
        # for it in range(infill_pixel_count):
        #     rand_row = np.random.randint(0,28)
        #     rand_col = np.random.randint(0,28)

        #     obstructed[:, :, rand_row, rand_col+1] = 257
        obstructed[:,:,15:25,15:25] = 257

        logits = net(obstructed)
        pred = torch.argmax(logits, dim=4)

        pred = pred[0, 0, :, 1:]
        im = im[0, :, 1:]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im)
        ax[1].imshow(pred)

        plt.show()

# Generation
# while True:
#     generated = net.sample(temp=0.2, batch_size=16)
#
#     fig, ax = plt.subplots(4, 4)
#     for i in range(4):
#         for j in range(4):
#             ax[i,j].axis('off')
#             ax[i,j].imshow(generated[i*4+j, 0, :, :].numpy())
#
#     plt.show()

# Infilling
# dat = MNISTImages(filepath="Datasets/mnist_test.csv")
# with torch.no_grad():
#     for im in dat:
#         im = im.view(1, 1, 28, 29)
#         im[:,:,15:25,15:25] = 0
#         missing_indices = np.full((1,1,28,29), fill_value=False)
#         missing_indices[:,:,15:20,15:20] = True
#
#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(im[0,0].numpy())
#         ax[1].imshow(net.infill(im,missing_indices, temp=2)[0,0].numpy())
#         plt.show()
