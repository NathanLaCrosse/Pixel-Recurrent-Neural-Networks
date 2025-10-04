import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def unpatch_image(im : torch.Tensor):
    num_patches, _ ,patch_dim = im.size()
    patch_dim = int(np.sqrt(patch_dim))
    unpatched = np.zeros((num_patches * patch_dim, num_patches * patch_dim))

    for row in range(num_patches):
        for col in range(num_patches):
            unpatched[patch_dim*row:patch_dim*(row+1), patch_dim*col:patch_dim*(col+1)] = im[row, col, :].numpy().reshape(patch_dim, patch_dim)

    return unpatched

net = na.TwoDimensionalGRUSeq2Seq(4, 7, 15, 14, 14, forcing=0)
net_dict = torch.load("LITEMonster1.pt", map_location=torch.device('cpu'))
net.load_state_dict(net_dict)
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

dat = md.PixelDataset(prc_len=14, filepath="Datasets/mnist_test.csv")

with torch.no_grad():
    for im, label in dat:
        pred, logvar, mean = net(im.view(1, 14, 14, 4))
        pred = pred[0]

        # latent = net.to_latent((im.view(1,14,14,4)))
        # print(latent)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(unpatch_image(im))
        ax[0].set_title("True")
        ax[1].imshow(unpatch_image(pred))
        # ax[1].imshow(unpatch_image(net.to_image(pred)))
        ax[1].set_title("Predicted")
        plt.show()