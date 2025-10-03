import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm

def unpatch_image(im : torch.Tensor):
    num_patches, _ ,patch_dim = im.size()
    patch_dim = int(np.sqrt(patch_dim))
    unpatched = np.zeros((num_patches * patch_dim, num_patches * patch_dim))

    for row in range(num_patches):
        for col in range(num_patches):
            unpatched[patch_dim*row:patch_dim*(row+1), patch_dim*col:patch_dim*(col+1)] = im[row, col, :].numpy().reshape(patch_dim, patch_dim)

    return unpatched

def generator():
    return

net = na.TwoDimensionalGRUSeq2Seq(4, 7, 15, 14, 14, forcing=0)
net_dict = torch.load("Models/LITEMonster10.pt", map_location=torch.device('cpu'))
net.load_state_dict(net_dict)
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

dat = md.PixelDataset(prc_len=14, filepath="Datasets/mnist_test.csv")

with torch.no_grad():
    for im, label in dat:
        pred = net(im.view(1, 14, 14, 4))[0]

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(unpatch_image(im))
        ax[0].set_title("True")
        ax[1].imshow(unpatch_image(pred))
        ax[1].set_title("Predicted")
        plt.show()


def train_generation(epochs = 1, batch_size = 256, learning_rate = 0.001, input_size = 16,
                    embedding_size = 32, hidden_size = 64, patch_rows = 7, patch_cols = 7,
                    latent_size = 64, num_layers = 1, forcing = 0.5, model_name = 'Omni.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    pixel_dataset = md.PixelDataset(prc_len=7)
    model = na.TwoDimensionalGRUSeq2Seq(input_size, embedding_size, hidden_size, patch_rows,
                                        patch_cols, latent_size, num_layers, forcing, device)
    model = model.to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        dat_loader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type==device))
        progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        model = model.train()
        for _, batch in enumerate(progress_bar):
            ims, label = batch
            ims = ims.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(ims)
              # Note on Loss Function: we want all images to be comprised of images with color values (-1, 1)
            # for tanh. however, we need values (0, 1) for Binary Cross Entropy loss
            loss = loss_fn(output, (ims + 1) / 2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})
            running_loss += loss.item()

        na.save_checkpoint(input_size, embedding_size, hidden_size, patch_rows,
                    patch_cols, latent_size, num_layers, forcing, model, model_name)