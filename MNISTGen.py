import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm
import torch.nn.functional as F

# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [64, 16]], which is output 0 of AsStridedBackward0, is at version 49; expected version 48 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
# torch.autograd.set_detect_anomaly(True)
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


dat = md.PixelDataset(prc_len=14)

net = na.TwoDimensionalGRUSeq2Seq(4, embedding_size=7, hidden_size=30, num_layers=1, forcing=0.5, device=device)
net = net.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(epochs):
    dat_loader = DataLoader(dat, batch_size=256, shuffle=True,  pin_memory=(device.type==device))
    progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    running_loss = 0.0
    net = net.train()
    for _, batch in enumerate(progress_bar):
        ims, label = batch

        ims = ims.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = net(ims)

        loss = loss_fn(output, ims)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.item()})
        running_loss += loss.item()

    torch.save(net.state_dict(), f"LITEMonster{epoch+1}.pt")