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
from NetworkArchitecture import RowRNN, OmniRowRNN, GenerativeRowRNN

epochs = 10
batch_size = 1
save = "Models/Generative.pt"

dat = md.PixelDataset(color=True, filepath="Datasets/Cartoons/Train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = GenerativeRowRNN(embed_size=64, hidden_size=64, num_layers=3, device=device)
net = net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(epochs):
    dat_loader = DataLoader(dat, batch_size=batch_size)
    progress_bar = tqdm.tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for _, batch in enumerate(progress_bar):
        ims = batch
        ims = ims.to(device)

        optimizer.zero_grad()
        logits = net(ims)

        loss = loss_fn(logits.reshape(-1, 257), ims[:, :, 1:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        progress_bar.set_postfix({"Loss:": loss.item()})

    torch.save(net.state_dict(), save)