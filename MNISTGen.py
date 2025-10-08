import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import NetworkArchitecture as na
import MNISTData as md
from tqdm import tqdm
import math

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
            pred, log_var, mean = model(im.view(1, patch_rows, patch_cols, patch_size))
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
                    pre_trained=None, beta_max=1):
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
    def loss_fn(x, pred, log_variance, mu, beta):
        reproduction_loss = nn.functional.binary_cross_entropy(x, pred, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp())
        return reproduction_loss, kl_loss, reproduction_loss + beta * kl_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        dat_loader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type==device))
        progress_bar = tqdm(dat_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        cycles_per_epoch = 0.5
        epoch_len = len(dat_loader)
        global_steps = 0
        steps_per_cycle = epoch_len // cycles_per_epoch

        model = model.train()
        for _, batch in enumerate(progress_bar):
            ims, label = batch
            ims = ims.to(device, non_blocking=True)

            optimizer.zero_grad()
            output, log_var, mean = model(ims)
            # Note on Loss Function: we want all images to be comprised of images with color values (-1, 1)
            # for tan_h. however, we need values (0, 1) for Binary Cross Entropy loss
            beta = beta_max * math.sin(math.pi * (global_steps % steps_per_cycle) / steps_per_cycle) ** 2
            rep_loss, kl_loss, loss = loss_fn(output, (ims + 1) / 2, log_var, mean, beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item(), 'beta' : beta, 'rep_loss' : rep_loss.item(), 'kl_loss' : kl_loss.item()})
            running_loss += loss.item()

            global_steps += 1

        na.save_checkpoint(input_size, embedding_size, hidden_size, patch_rows,
                    patch_cols, latent_size, num_layers, forcing, model, model_file_name)

def test_centering(filepath="VAE.pt", device=torch.device("cpu")):
    """
    Visualize whether or not there is a fuzzy region around each of the means in the VAE encoding.
    Essentially, we generate a bunch of vectors close to the mean and see what they look like
    """

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

    per_row = 10
    scales = [1, 10, 100, 1000]

    with torch.no_grad():
        for im, label in eval_dataset:
            pred, log_var, mean = model(im.view(1, patch_rows, patch_cols, patch_size))
            pred = pred[0]
            mean = mean[0]
            log_var = log_var[0]

            fig, ax = plt.subplots(nrows=len(scales), ncols=per_row)
            for i in range(len(scales)):
                for k in range(per_row):
                    ax[i, k].axis('off')
                    ax[i, k].imshow(unpatch_image(model.sample_image(log_var * scales[i], mean)[0], patch_rows, patch_cols, patch_size))

            plt.tight_layout()
            plt.show()

def interp(filepath="VAE.pt", device=torch.device("cpu")):
    """
    Sample two latent vectors and visualize how one turns into the other.
    """
    checkpoint = torch.load(f'Models/{filepath}', map_location=device)

    config = checkpoint['config']
    model = na.TwoDimensionalGRUSeq2Seq(**config)
    eval_dataset = md.PixelDataset(filepath="Datasets/mnist_test.csv", prc_len=config['patch_rows'])
    model.load_state_dict(checkpoint['model'])
    model.forcing = 0.0

    patch_rows = config['patch_rows']
    patch_cols = config['patch_cols']
    patch_size = config['input_size']

    with torch.no_grad():
        for k in range(len(eval_dataset)-1):
            im1, _ = eval_dataset[k]
            im2, _ = eval_dataset[k+1]

            _, s, mean1 = model(im1.view(1, patch_rows, patch_cols, patch_size))
            _, _, mean2 = model(im2.view(1, patch_rows, patch_cols, patch_size))
            s = s[0]
            mean1 = mean1[0]
            mean2 = mean2[0]

            slope = mean2 - mean1
            steps = 20

            fig, ax = plt.subplots(1, steps)

            for i in range(steps):
                delta = slope * i / steps

                ax[i].axis('off')
                ax[i].imshow(unpatch_image(model.forward(mean1 + delta, just_decoder=True, reparameterize=False)[0], patch_rows, patch_cols, patch_size))

            plt.show()

def sample_space(filepath="VAE.pt", device=torch.device("cpu")):
    checkpoint = torch.load(f'Models/{filepath}', map_location=device)

    config = checkpoint['config']
    model = na.TwoDimensionalGRUSeq2Seq(**config)
    model.load_state_dict(checkpoint['model'])
    model.forcing = 0.0

    patch_rows = config['patch_rows']
    patch_cols = config['patch_cols']
    patch_size = config['input_size']

    pixel_dataset = md.PixelDataset(prc_len=patch_rows)

    with torch.no_grad():
        # Calculate global mean & global standard deviation
        mean = torch.zeros(config['latent_size'])
        std = torch.zeros(config['latent_size'])

        dat_loader = DataLoader(pixel_dataset, 2000, shuffle=False)
        progress = tqdm(dat_loader, desc="Calculating mean & std")
        for _, data in enumerate(progress):
            ims, labels = data

            _, log_vars, means = model(ims)

            mean = mean + torch.sum(means, dim=0)
            std = std + torch.sum(torch.exp(log_vars/2), dim=0)

        mean = mean / len(pixel_dataset)
        std = std / len(pixel_dataset)

        # Generate a 5 x 5 grid of samples based off of global mean & standard deviation
        while True:
            dim = 7
            fig, ax = plt.subplots(dim, dim)
            for i in range(dim):
                for j in range(dim):
                    samp = mean + std * torch.randn(config['latent_size'])

                    im = model.forward(samp, just_decoder=True, reparameterize=False)

                    ax[i, j].axis('off')
                    ax[i, j].imshow(unpatch_image(im[0], patch_rows, patch_cols, patch_size))

            plt.show()

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loaded = na.load_checkpoint(filepath="RefinedVAE.pt", device=device)
    # train_generation(50, batch_size=2048, learning_rate=0.0001, input_size=16, embedding_size=20, hidden_size=50,
    #                  patch_rows=7, patch_cols=7, latent_size=40, num_layers=1, forcing=0.95, model_file_name="VAE5.pt", pre_trained=loaded, beta_max=10)
 
    # generator(filepath="VAE_dif_dim.pt")

    # generator(filepath="VAE4.pt")
    # test_centering(filepath="VAE2.pt")
    # interp(filepath="VAE3.pt")
    sample_space(filepath="RefinedVAE2.pt")