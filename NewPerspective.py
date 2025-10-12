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


class RowRNN(nn.Module):
    def __init__(self, embed_size=64, hidden_size=32, num_layers=3):
        super(RowRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(257, embed_size)
        self.gru = nn.GRU(input_size=embed_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.to_out = nn.Linear(hidden_size, 257)

    def forward(self, x):
        batch_size, channels, rows, cols = x.size()

        # Add row signature to the row
        x = x.view(batch_size, rows, cols)
        x = self.embedding(x) # Now size is batch_size x rows x cols x embed_size
        row_data = torch.arange(0, rows) / rows * 2 - 1
        row_data = row_data.repeat(batch_size, 1, cols, 1).permute(0, 3, 2, 1)
        x = torch.cat((x, row_data), dim=3)
        x = x.view(batch_size*rows, cols, self.embed_size+1)

        # Consider each row, across all batches, separately
        # x = x.view(batch_size * rows, cols)
        # x = self.embedding(x)
        # x = x.view(batch_size*rows, cols, self.embed_size)

        # Gather all of the hidden vectors
        outputs, _ = self.gru(x)

        # Apply conversion to pixel intensities
        pred = self.to_out(outputs)
        return pred.view(batch_size, channels, rows, cols, 257)


dat = MNISTImages()

epochs = 5
batch_size = 128

# ---------- Training Code ----------
# net = RowRNN()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#
# for epoch in range(epochs):
#     dat_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
#     progress_bar = tqdm.tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
#     running_loss = 0.0
#
#     for _, batch in enumerate(progress_bar):
#         ims = batch
#         # hot_encoded = F.one_hot(torch.tensor(ims, dtype=torch.long), 257)
#
#         optimizer.zero_grad()
#         logits = net(ims) # Result: batch_size, 1, 28, 29, 257
#
#         # Clip out start-of-sequence blip
#         logits = logits[:,:,:,1:,:]
#         ims = ims[:,:,:,1:]
#
#         loss = loss_fn(logits.reshape(-1, 257), ims.reshape(-1))
#         loss.backward()
#         nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
#         optimizer.step()
#
#         running_loss += loss.item()
#         progress_bar.set_postfix({"Loss:": loss.item()})
#
#     torch.save(net.state_dict(), "RowRNN.pt")

# ---------- Testing Code ----------
dat = MNISTImages(filepath="Datasets/mnist_test.csv")
net = RowRNN()
state_dict = torch.load("RowRNN.pt")
net.load_state_dict(state_dict)
net.eval()

with torch.no_grad():
    for im in dat:
        logits = net(im.view(1, 1, 28, 29))
        pred = torch.argmax(logits, dim=4)

        pred = pred[0, 0, :, 1:]
        im = im[0, :, 1:]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im)
        ax[1].imshow(pred)

        plt.show()


