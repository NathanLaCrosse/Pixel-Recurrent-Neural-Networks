import torch
import pandas as pd
import tqdm as tqdm
from torch.utils.data import Dataset

# MNIST Images are 28 x 28 pixels
def load_mnist(filepath, samples=60000):
    mnist_train = pd.read_csv(filepath)
    labels = []
    images = []

    progress_bar = tqdm.tqdm(range(len(mnist_train)), desc="Loading Dataset")
    its = 0
    for _, row in enumerate(progress_bar):
        labels.append(mnist_train.iloc[row, 0])
        images.append(mnist_train.iloc[row, 1:].to_numpy().reshape((28,28)))

        its += 1
        if its > samples:
            break

    return labels, images

def to_patches(image, prc_len):
    """
    Assumptions: Example Image is always square, Image is grayscale
    prc_len: Length of patch rows and columns
    patch_size: Number of pixels per patch
    """

    patch_len = int(len(image[0]) / prc_len)
    patch_size = int(patch_len ** 2)

    image = image / 255 * 2 - 1
    patched_image = image.reshape(prc_len, patch_len, prc_len, patch_len)
    patched_image = patched_image.transpose(0, 2, 1, 3)
    patched_image = patched_image.reshape(prc_len, prc_len, patch_size)
    patched_tensor = torch.tensor(patched_image, dtype=torch.float32)
    return patched_tensor


class PixelDataset(Dataset):

    def __init__(self, filepath = 'Datasets/mnist_train.csv', prc_len = 7, samples=1000000):
        labels, images = load_mnist(filepath, samples=samples)
        self.labels = labels
        self.patched_images = [to_patches(im, prc_len) for im in tqdm.tqdm(images)]
        self.raw_images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patched_images[idx], self.labels[idx]

