import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm

def unpatch_image(im, patch_rows, patch_cols, patch_size):
    patch_dim = int(np.sqrt(patch_size))

    im = im.numpy()
    image = im.reshape(patch_rows, patch_cols, patch_dim, patch_dim)
    image = image.transpose(0, 2, 1, 3)
    image = image.reshape(28, 28)

    return image

def generator(filepath, device=torch.device("cpu")):
    checkpoint = torch.load(f'Models/{filepath}', map_location=device)

    config = checkpoint['config']
    model = na.TwoDimensionalGRUSeq2Seq(**config)
    eval_dataset = md.PixelDataset(filepath="Datasets/mnist_test.csv", prc_len=config['patch_rows'])
    model.load_state_dict(checkpoint['model'])
    model.forcing = 0.0

    patch_rows = config['patch_rows']
    patch_cols = config['patch_cols']
    patch_size = config['input_size']

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    with torch.no_grad():
        for im, label in eval_dataset:
            pred, _, _ = model(im.view(1, patch_rows, patch_cols, patch_size))
            pred = pred[0]

            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(unpatch_image(im, patch_rows, patch_cols, patch_size), cmap = 'gray')
            ax[0].set_title("True")
            ax[1].imshow(unpatch_image(pred, patch_rows, patch_cols, patch_size), cmap = 'gray')
            ax[1].set_title("Predicted")
            plt.show()




def train_generation(epochs = 1, batch_size = 256, learning_rate = 0.001, input_size = 16,
                    embedding_size = 32, hidden_size = 64, patch_rows = 7, patch_cols = 7,
                    latent_size = 64, num_layers = 1, forcing = 0.5, model_file_name = 'Omni.pt',
                    pre_trained=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    pixel_dataset = md.PixelDataset(prc_len=patch_rows)
    if pre_trained is None:
        model = na.TwoDimensionalGRUSeq2Seq(input_size, embedding_size, hidden_size, patch_rows,
                                        patch_cols, latent_size, num_layers, forcing, device)
    else:
        model = pre_trained
    model = model.to(device, non_blocking=True)

    # loss_fn = nn.BCELoss()
    def loss_fn(x, pred, log_variance, mu):
        reproduction_loss = nn.functional.binary_cross_entropy(x, pred, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
        return reproduction_loss + kl_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        dat_loader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type==device))
        progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        model = model.train()
        for _, batch in enumerate(progress_bar):
            ims, label = batch
            ims = ims.to(device, non_blocking=True)

            optimizer.zero_grad()
            output, log_var, mean = model(ims)
            # Note on Loss Function: we want all images to be comprised of images with color values (-1, 1)
            # for tan_h. however, we need values (0, 1) for Binary Cross Entropy loss
            loss = loss_fn(output, (ims + 1) / 2, log_var, mean)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})
            running_loss += loss.item()

        na.save_checkpoint(input_size, embedding_size, hidden_size, patch_rows,
                    patch_cols, latent_size, num_layers, forcing, model, model_file_name)