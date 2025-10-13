import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.xpu import device

import NetworkArchitecture as na
import MNISTData as md
import MNISTGen as mg
from tqdm import tqdm
import math

patch_size = 2
patch_pixels = patch_size**2
forcing_reduction_rate = 0.005
min_forcing = 0.9

# ----------------- Training Code -----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = na.GenerativeTwoDimensionalGRU(patch_pixels, 64, 50, forcing=1.0, device=device)

dat = md.MNISTPixelDataset(prc_len=28//patch_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net = net.to(device)

epochs = 15
batch_size = 512

for epoch in range(epochs):
    dat_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
    progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    running_loss = 0.0

    for _, batch in enumerate(progress_bar):
        ims, labels = batch

        optimizer.zero_grad()
        logits = net(ims.view(len(ims), 28//patch_size, 28//patch_size, patch_pixels).to(device))

        loss = loss_fn(logits.permute(0, 4, 1, 2, 3), labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss:" : loss.item()})

    net.forcing = max(min_forcing, net.forcing - forcing_reduction_rate)
    torch.save(net.state_dict(), "Models/GenPixelClf.pt")

#     if (epoch+1) % 20 == 0:
#         torch.save(net.state_dict, f"{(epoch+1)//20}GenPixelClf.pt")

# ----------------- Testing Code -----------------
# device = torch.device('cpu')
# net = na.GenerativeTwoDimensionalGRU(patch_size, 64, 50, forcing=0.9, device=device)
# state_dict = torch.load("GenPixelClf.pt", map_location=torch.device('cpu'))
# net.load_state_dict(state_dict)

# dat = md.MNISTPixelDataset(prc_len=28//patch_size, filepath="Datasets/mnist_test.csv")

# for im, label in dat:
#     logits = net(im.view(1, 28//patch_size, 28//patch_size, patch_pixels))
#     vals = torch.argmax(logits, dim=4)

#     unpatched = mg.unpatch_image(vals, 28//patch_size, 28//patch_size, patch_pixels)

#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(mg.unpatch_image(im, 28//patch_size, 28//patch_size, patch_pixels), cmap='gray')
#     ax[1].imshow(unpatched, cmap='gray')
#     plt.show()