import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm
import torch.nn.functional as F

class TwoDGRUClassifier(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, classification_count):
        super(TwoDGRUClassifier, self).__init__()

        self.recurrent = na.TwoDimensionalGRU(input_size=input_size, embedding_size=embed_size, hidden_size=hidden_size)
        self.to_out = nn.Linear(hidden_size, classification_count)

    def forward(self, x):
        h_n = self.recurrent(x)
        return self.to_out(h_n)


epochs = 5

dat = md.PixelDataset()
dat_loader = DataLoader(dat, batch_size=16, shuffle=True)

net = TwoDGRUClassifier(16, embed_size=20, hidden_size=50, classification_count=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

progress_bar = tqdm(dat_loader, desc=f"Epoch {0 + 1}/{epochs}")
running_loss = 0.0
for _, batch in enumerate(progress_bar):
    ims, label = batch

    optimizer.zero_grad()
    output = net(ims)

    loss = loss_fn(output, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # Gradient clipping
    optimizer.step()

    progress_bar.set_postfix({'loss': loss.item()})
    running_loss += loss.item()

    progress_bar.set_description(desc=f"Loss: {loss}")

# Test performance
evil_dat = md.PixelDataset(filepath="Datasets/mnist_test.csv")

# net = net.eval()
with torch.no_grad():
    correct = 0
    for im, label in evil_dat:
        logits = net(im.view(1, 7, 7, 16))

        max_dex = torch.argmax(logits).item()

        if max_dex == label:
            correct += 1

    correct = correct / len(evil_dat)

    print(f"Test Accuracy: {correct}")
