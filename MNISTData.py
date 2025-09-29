import numpy as np
import torch
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# For Testing
def show(im):
    plt.imshow(im, cmap='grey')
    plt.show()

  # MNIST Images are 28 x 28 pixels
def load_mnist(filepath, samples=1000000000):
    mnist_train = pd.read_csv(filepath)
    labels = []
    images = []

    # I found a faster way ^_^
    # mnist_train is a DataFrame object - the .iloc method lets us manipulate it like a numpy
    # array - and then we can just turn it into a numpy array and change its view (~O(1) time)
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
    Patch_Size: Number of pixels per patch
    """

    patch_len = int(len(image[0]) / prc_len)
    patch_size = int(patch_len**2)
    patched_image = np.empty((prc_len, prc_len, patch_size))

    for i in range(prc_len):
        for j in range(prc_len):
            patch = np.array(image[patch_len * i: patch_len * (i + 1),
                                        patch_len * j: patch_len * (j + 1)])
            patched_image[i][j] = 2 * (patch.flatten() / 255) - 1
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

if __name__ == "__main__":
    pixel = PixelDataset('Datasets/mnist_train.csv', 7)

    # im = pixel.raw_images[0]
    # show(im)