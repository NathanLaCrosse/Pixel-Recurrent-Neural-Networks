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
from NetworkArchitecture import ConditionalRowRNN


def train_model(training_args):
    """
    Trains an infill network to infill randomly removed parts of an image, which are labeled
    with the intensity 257. Saves to a pt file every epoch.

    :param training_args: A dictionary containing the different settings of the trainer, which are as follows:
        - epochs: Number of epochs
        - im_rows: Number of rows in the image. Images are assumed to be square
        - net: The neural network to train
        - dat: The dataset object
        - file_name: The file to save the weights to
        - device: The device to train on
        - starting_infill_pixels: The starting count of pixels to convert to an infill token
        - infill_pixel_increment: The amount of infill pixels increases at this rate every epoch
        - start_infill_grid: The initial size of an infill pixel
        - max_infill_grid: The maximum size of an infill pixel
        - epochs_per_grid_increment: The number of epochs to increment infill grid size
        - max_infill_pixels: The maximum number of pixels to infill
        - infill_level_probs: The probability to infill over the RGB, GB and B channels respectively
    :return:
    """

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
            if isinstance(net, ConditionalRowRNN):
                logits = net(obstructed, target=ims.to(device))  # Result: batch_size, 1, size, size + 1, 258
            else:
                logits = net(obstructed)

            # Clip out start-of-sequence blip - not needed for grading
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


def perform_infill_with_temperature(net, im : torch.Tensor, mask : np.ndarray,
                                    temp_pixels=10, start_temp=0.8, temp_dec_rate=0.05, min_temp=0.1):
    """

    :param net:
    :param im:
    :param mask:
    :param temp_pixels:
    :param start_temp:
    :param temp_dec_rate:
    :param min_temp:
    :return:
    """

    with torch.no_grad():
        temp = start_temp
        reconstructed = torch.clone(im.view(1, 3, mask.shape[0] + 1, mask.shape[1] + 1))

        painted_pixels = 0

        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                if mask[row, col] and painted_pixels < temp_pixels:
                    prompt = torch.clone(reconstructed)
                    prompt[:, :, row+1, col+1] = 257

                    pred = net(prompt, temp=temp)

                    reconstructed[:, :, row+1, col+1] = pred[:, :, row+1, col+1]

                    temp = max(min_temp, temp - temp_dec_rate)
                    painted_pixels += 1
                elif mask[row, col]:
                    prompt[:, :, row+1, col+1] = 257

        logits = net(reconstructed)
        pred = torch.argmax(logits, dim=4)

        pred = pred[0, :, 1:, 1:]
        pred = pred.permute(1, 2, 0)

        return pred




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dat = md.PixelDataset(color=True, filepath="Datasets/Cartoons/Train")

device = torch.device("cpu")
dat = md.PixelDataset(color=True, filepath="Datasets/Cartoons/Test")

training_args = {
    "epochs" : 100,
    "batch_size" : 1,
    "im_rows" : 36,
    "net" : ConditionalRowRNN(embed_size=64, hidden_size=96, num_layers=5, channels=3, device=device),
    "dat" : dat,
    "file_name" : "Models/NewInfill.pt",
    "device" : device,
    "starting_infill_pixels" : 10,
    "infill_pixel_increment" : 10,
    "start_infill_grid" : 1,
    "max_infill_grid" : 6,
    "epochs_per_grid_increment" : 4,
    "max_infill_pixels" : 36 * 36 * 0.6,
    "infill_level_probs" : [0.5, 0.25, 0.25]
}
# train_model(training_args)

# ---------- Testing Code ----------

net = training_args["net"]
state_dict = torch.load("Models/NewInfillAdamGood.pt", map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.eval()

with torch.no_grad():
    for im in dat:
        obstructed = torch.clone(im.view(1, 3, 37, 37))
        display_im = im.permute(1, 2, 0).numpy()[1:,1:,:]
        unchanged = display_im.copy()

        mask = np.full((36,36), fill_value=False)
        mask[15:20, 15:25] = True

        # pred = perform_infill_with_temperature(net, im, mask, start_temp=1.0, temp_pixels=20)
        for c in range(3):
            obstructed[0, c][1:,1:][mask] = 257
        display_im[mask] = np.array([0, 255, 0])

        # Convert logits to an image - generate prediction
        logits = net(obstructed)
        pred = torch.argmax(logits, dim=4)
        pred = pred[0, :, 1:, 1:]
        pred = pred.permute(1, 2, 0)

        fig, ax = plt.subplots(2, 2)

        for i, j in np.ndindex((2,2)):
            ax[i, j].axis('off')

        # Display the unchanged and masked images
        ax[0, 0].imshow(unchanged)
        ax[0, 0].set_title("Original")
        ax[0, 1].imshow(display_im)
        ax[0, 1].set_title("Original w/ Mask")

        # Display a zero temperature and slightly warm prediction
        ax[1, 0].imshow(pred)
        ax[1, 0].set_title("Without Temperature")
        temp_pixels = 10
        ax[1, 1].imshow(perform_infill_with_temperature(net, obstructed, mask, temp_pixels=temp_pixels, start_temp=1))
        ax[1, 1].set_title(f"With {temp_pixels} Temperature Steps")

        plt.show()
