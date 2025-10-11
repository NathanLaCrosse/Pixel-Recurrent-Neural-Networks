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

# ----------------- Training Code -----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = 1
patch_pixels = patch_size**2

net = na.GenerativeTwoDimensionalGRU(patch_pixels, 64, 25, forcing=1.0, device=device)

dat = md.MNISTPixelDataset(prc_len=28)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net = net.to(device)

epochs = 20
batch_size = 64

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

    torch.save(net.state_dict(), "GenPixelClf.pt")

#     if (epoch+1) % 10 == 0:
#         torch.save(net.state_dict, f"{(epoch+1)//10}GenPixelClf.pt")

# ----------------- Testing Code -----------------
# device = torch.device('cpu')
# net = na.GenerativeTwoDimensionalGRU(4, 64, 50, forcing=1.0, device=device)
# state_dict = torch.load("GenPixelClf.pt", map_location=torch.device('cpu'))
# net.load_state_dict(state_dict)
#
# dat = md.MNISTPixelDataset(prc_len=14, filepath="Datasets/mnist_test.csv")
#
# for im, label in dat:
#     logits = net(im.view(1, 14, 14, 4))
#     vals = torch.argmax(logits, 4)
#
#     unpatched = mg.unpatch_image(vals, 14, 14, 4)
#
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(mg.unpatch_image(im, 14, 14, 4), cmap='gray')
#     ax[1].imshow(unpatched, cmap='gray')
#     plt.show()