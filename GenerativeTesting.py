import os

import numpy
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
from NetworkArchitecture import RowRNN, OmniRowRNN, GenerativeRowRNN

save = "Models/Generative.pt"
dat = md.PixelDataset(color=True, filepath="Datasets/Cartoons/Test")
device = torch.device("cpu")
net = GenerativeRowRNN(embed_size=64, hidden_size=64, num_layers=1, device=device)

state_dict = torch.load("Models/GenerativeAdam1.pt", map_location=device)
net.load_state_dict(state_dict)

net = net.eval()

with torch.no_grad():
    for im in dat:
        im = im.view(1, 3, 37, 37)

        mask = np.full((1, 3, 37, 37), fill_value=False)
        # mask[:,:,15:25,15:25] = True
        mask[:,:,3:,18:] = True

        pred = net.predict(im, mask, temp=1)[0]
        pred = pred.permute(1, 2, 0)

        im = im[0].permute(1, 2, 0)
        mask = mask[0].astype(np.uint8) * 255
        mask = numpy.transpose(mask, (1, 2, 0))

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(im)
        ax[1].imshow(mask)
        ax[2].imshow(pred)

        plt.show()