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
from NetworkArchitecture import RowRNN, OmniRowRNN, AltRowRNN


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




# ---------- Training Code ----------
def train_model(training_args):
    net = training_args["net"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net = net.to(device=device)

    infill_pixel_count = training_args["starting_infill_pixels"]
    current_grid_max = training_args["start_infill_grid"]

    for epoch in range(training_args["epochs"]):
        dat_loader = DataLoader(training_args["dat"], batch_size=training_args["batch_size"], shuffle=True)
        progress_bar = tqdm.tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{training_args["epochs"]}")
        running_loss = 0.0

        for _, batch in enumerate(progress_bar):
            ims = batch

            # Obstruct random pixel blocks from the image
            remaining_infill = infill_pixel_count
            obstructed = torch.clone(ims)
            while remaining_infill > 0:
                block_size = np.random.randint(1, current_grid_max + 1)

                rand_row = np.random.randint(0, training_args["im_rows"])
                rand_col = np.random.randint(0, training_args["im_rows"])

                mask_lvl = np.random.choice([0, 1, 2], p=training_args["infill_level_probs"])

                # Apply a random mask that's conditional on the channels
                obstructed[:, mask_lvl:, rand_row:rand_row + block_size, rand_col + 1:rand_col + 1 + block_size] = 257

                remaining_infill -= block_size ** 2

            obstructed = obstructed.to(device)

            optimizer.zero_grad()
            logits = net(obstructed, target=ims.to(device))  # Result: batch_size, 1, size, size + 1, 258

            # Clip out start-of-sequence blip
            logits = logits[:, :, 1:, 1:, :]
            ims = ims[:, :, 1:, 1:]

            loss = loss_fn(logits.reshape(-1, 258), ims.reshape(-1).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss:": loss.item()})

        infill_pixel_count += training_args["infill_pixel_increment"]
        infill_pixel_count = min(infill_pixel_count, training_args["max_infill_pixels"])

        if (epoch + 1) % training_args["epochs_per_grid_increment"] == 0:
            current_grid_max = min(current_grid_max + 1, training_args["max_infill_grid"])

        torch.save(net.state_dict(), training_args["file_name"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

training_args = {
    "epochs" : 100,
    "batch_size" : 100,
    "im_rows" : 36,
    "net" : AltRowRNN(embed_size=40, hidden_size=100, num_layers=5, channels=3, device=device),
    "dat" : md.PixelDataset(color=True, filepath="Datasets/Cartoons/Train"),
    "file_name" : "Models/NewInfill.pt",
    "device" : device,
    "starting_infill_pixels" : 10,
    "infill_pixel_increment" : 10,
    "start_infill_grid" : 1,
    "max_infill_grid" : 6,
    "epochs_per_grid_increment" : 4,
    "max_infill_pixels" : 36 * 36 * 0.6,
    "infill_level_probs" : [0.20, 0.35, 0.45]
}
train_model(training_args)

# ---------- Testing Code ----------

# net = training_args["net"]
# state_dict = torch.load("Models/AltInfill.pt", map_location=torch.device('cpu'))
# net.load_state_dict(state_dict)
# net.eval()

# grid_size = 36
# infill_pixel_count = 10

# # Classic reconstruction. (Sanity Check)
# dat = md.PixelDataset(filepath="Datasets/Cartoons/Test", color=True)
# with torch.no_grad():
#     for im in dat:
#         obstructed = im.view(1, 3, grid_size, grid_size+1)
#         for _ in range(infill_pixel_count):
#             rand_row = np.random.randint(0,grid_size)
#             rand_col = np.random.randint(0,grid_size)

#             obstructed[:, :, rand_row:rand_row+3, rand_col+1:rand_col+4] = 257

#         # obstructed[:, :, 3:, 18:] = 257

#         # obstructed[:,:,15:25,15:25] = 257

#         # Convert logits to an image
#         logits = net(obstructed)
#         pred = torch.argmax(logits, dim=4)
#         pred = pred[0, :, :, 1:]
#         im = im.view(1, 3, grid_size, grid_size + 1)[0, :, :, 1:]
#         pred = pred.permute(1, 2, 0)
#         im = im.permute(1, 2, 0).clamp(0, 255)

#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(im)
#         ax[1].imshow(pred)

#         # # Generate some samples to look at
#         # for s in range(4):
#         #     samp = generate_with_temperature(net, obstructed, 2)
#         #     ax[1, s].imshow(samp[:, 1:, :])

#         plt.show()
