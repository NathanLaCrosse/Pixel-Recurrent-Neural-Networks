import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm
import math

net = na.ClassifyingUnderlyingTwoDimensionalGRU(4, 16, 30)

dat = md.MNISTPixelDataset(prc_len=14)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 5
batch_size = 128

for epoch in range(epochs):
    dat_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
    progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    running_loss = 0.0

    for _, batch in enumerate(progress_bar):
        ims, labels = batch

        optimizer.zero_grad()
        logits, output, h_final = net(ims.view(len(ims), 14, 14, 4))

        loss = loss_fn(logits[0], labels.to(torch.float32))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss:" : loss.item()})

    torch.save(net.state_dict(), "PixelClf.pt")