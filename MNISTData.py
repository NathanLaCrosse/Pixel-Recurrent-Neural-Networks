import os
import torch
import pandas as pd
import tqdm as tqdm
from torch.utils.data import Dataset
import cv2

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


        # im = torch.tensor(self.raw_images[item], dtype=torch.long)
        # start_of_col = torch.ones((28, 1), dtype=torch.long) * 256
        # im = torch.cat((start_of_col, im), dim=1)
        # return im.view(1, 28, 29)

def process_image(path, size, color):
    if color:
        image = cv2.imread(path)[100:400, 100:400, ::-1]
        image = torch.tensor(cv2.resize(image, (size, size)), dtype=torch.long)
        start_of_col = torch.ones((size, 1, 3), dtype=torch.long) * 256
        start_of_row = torch.ones((1, size+1, 3), dtype=torch.long) * 256
        image = torch.cat((start_of_col, image), dim=1)
        image = torch.cat((start_of_row, image), dim=0)
        return image.permute(2, 0, 1)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[100:400, 100:400]
        image =  torch.tensor(cv2.resize(image, (size, size)), dtype=torch.long)
        start_of_col = torch.ones((size, 1), dtype=torch.long) * 256
        image = torch.cat((start_of_col, image), dim=1)
        return image.view(1, size, size + 1)


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

    def __init__(self, filepath = 'Datasets/Cartoons', resize = 36, color = False, samples=1000000):
        self.directories = generate_directory_list(filepath, samples)
        self.resize = resize
        self.color = color

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, idx):
        image = self.directories[idx]
        return process_image(image, self.resize, self.color)


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