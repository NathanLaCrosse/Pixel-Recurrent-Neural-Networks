import os
import torch
import pandas as pd
import tqdm as tqdm
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F

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

def to_patches(image, prc_len, one_hot):
    """
    Assumptions: Example Image is always square, Image is grayscale
    prc_len: Length of patch rows and columns
    patch_size: Number of pixels per patch
    """

    patch_len = int(len(image[0]) / prc_len)
    patch_size = int(patch_len ** 2)

    patched_image = image.reshape(prc_len, patch_len, prc_len, patch_len)
    patched_image = patched_image.transpose(0, 2, 1, 3)
    patched_image = patched_image.reshape(prc_len, prc_len, patch_size)
    if one_hot:
        # return F.one_hot(torch.tensor(patched_image, dtype=torch.long), 256)
        return torch.tensor(patched_image, dtype=torch.long)
    else:
        return torch.tensor((patched_image / 255) * 2 - 1, dtype=torch.float32)


def process_image(path, size, prc_len):
    im = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size, size))
    im = to_patches(im, prc_len, False)
    return im


def generate_directory_list(filepath, samples):
    directories = []
    for root, dirs, files in os.walk(filepath):
        for file in tqdm.tqdm(files):
            path = os.path.join(root, file)
            path = path.replace("/", "\\")
            directories.append(path)
            if len(directories) > samples:
                break
    return directories


class PixelDataset(Dataset):

    def __init__(self, filepath = 'Datasets/Test', prc_len = 6, resize = 36, samples=1000000):
        self.directories = generate_directory_list(filepath, samples)
        self.resize = resize
        self.prc_len = prc_len

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        image = self.directories[idx]
        return process_image(image, self.resize, self.prc_len), None


class MNISTPixelDataset(Dataset):

    def __init__(self, filepath = 'Datasets/mnist_train.csv', prc_len = 7, samples=1000000):
        labels, images = load_mnist(filepath, samples=samples)
        self.prc_len = prc_len
        self.labels = labels
        self.raw_images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (to_patches(self.raw_images[idx], self.prc_len, False),
                to_patches(self.raw_images[idx], self.prc_len, True))