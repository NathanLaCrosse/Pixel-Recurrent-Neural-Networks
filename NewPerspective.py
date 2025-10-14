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
from NetworkArchitecture import RowRNN, FastRowRNN

class MNISTImages(Dataset):
    def __init__(self,filepath="Datasets/mnist_train.csv", samples=60000):
        _, self.raw_images = md.load_mnist(filepath=filepath, samples=samples)

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, item):
        im = torch.tensor(self.raw_images[item], dtype=torch.long)
        start_of_col = torch.ones((28, 1), dtype=torch.long) * 256
        im = torch.cat((start_of_col, im), dim=1)
        return im.view(1, 28, 29)


def train_infill_model(epochs, batch_size, embed_size, hidden_size, numlayers, color=False, save_file="InfillRNN.pt",
                       infill_pixel_count=3, infill_increment=3, infill_grid_max=4, current_grid_max=1, epochs_per_grid_increment=10, size = 36):

    dat = md.PixelDataset(color=color, filepath="Datasets/Cartoons/Train")
    max_infill_pixels = 0.2 * size**2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RowRNN(embed_size=embed_size, hidden_size=hidden_size, num_layers=numlayers, channels=3 if color else 1, device=device)
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
                
                rand_row = np.random.randint(0,size)
                rand_col = np.random.randint(0,size)

                obstructed[:, :, rand_row:rand_row+block_size, rand_col+1:rand_col+1+block_size] = 257
                remaining_infill -= block_size**2

            obstructed = obstructed.to(device)

            optimizer.zero_grad()
            logits = net(obstructed) # Result: batch_size, 1, size, size + 1, 258

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

        if (epoch + 1) % epochs_per_grid_increment == 0:
            current_grid_max = min(current_grid_max + 1, infill_grid_max)

        torch.save(net.state_dict(), save_file)
    
    return running_loss

# ---------- Training Code ----------
epochs = 100
batch_size = 64
im_rows = 36

infill_pixel_count = 10
infill_increment = 15
infill_grid_max = 5
epochs_per_grid_increment = 3
current_grid_max = 1
max_infill_pixels = im_rows * im_rows * 0.5

# train_infill_model(epochs, batch_size, embed_size=64, hidden_size=96, numlayers=5, color=True, save_file="Models/ChannelInfill.pt",
#                    infill_pixel_count=infill_pixel_count, infill_increment=infill_increment, infill_grid_max=infill_grid_max,
#                    current_grid_max=current_grid_max, epochs_per_grid_increment=epochs_per_grid_increment, size=im_rows)

# ---------- Testing Code ----------
net = RowRNN(embed_size=64, hidden_size=96, num_layers=5, channels=3)
# net = RowRNN(embed_size=64, hidden_size=128, num_layers=10)
state_dict = torch.load("Models/FaceInfill1.pt", map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()

grid_size = 36
infill_pixel_count = 10

# Classic reconstruction. (Sanity Check)
# dat = MNISTImages(filepath="Datasets/mnist_test.csv")
dat = md.PixelDataset(filepath="Datasets/Cartoons/Test", color=True)
with torch.no_grad():
    for im in dat:
        obstructed = im.view(1, 3, grid_size, grid_size+1)
        # for _ in range(infill_pixel_count):
        #     rand_row = np.random.randint(0,grid_size)
        #     rand_col = np.random.randint(0,grid_size)

        #     obstructed[:, :, rand_row:rand_row+4, rand_col+1:rand_col+5] = 257

        obstructed[:, :, 18:, 3:] = 257

        # obstructed[:,:,15:25,15:25] = 257

        logits = net(obstructed)
        pred = torch.argmax(logits, dim=4)

        pred = pred[0, :, :, 1:]
        im = im.view(1, 3, grid_size, grid_size+1)[0, :, :, 1:]

        pred = pred.permute(1, 2, 0)
        im = im.permute(1, 2, 0)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im)
        ax[1].imshow(pred)

        plt.show()
