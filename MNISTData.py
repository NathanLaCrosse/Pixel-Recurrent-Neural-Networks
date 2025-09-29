import numpy as np
import torch
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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

    # it = 0
    # for _, row in tqdm.tqdm(mnist_train.iterrows()):
    #     pict = np.zeros((28, 28), dtype=np.float32)
    #     for j in range(28):
    #         for k in range(28):
    #             pict[j, k] = row[f'{j + 1}x{k + 1}']
    #     labels.append(row['label'])
    #     images.append(pict)
    #
    #     it+=1
    #     if it >= max_loaded:
    #         break

    return labels, images

'''
Dataloader: (Tentative: 7, 7, 16)
Patch_Rows: How many patches per row
Patch_Cols: How many patches per column
Patch_Size: Number of pixels per patch 
'''


def to_patches(image):
    patched_image = np.empty((7, 7, 16))
    for i in range(7):
        for j in range(7):
            patch = np.array(image[4 * i: 4 * (i + 1), 4 * j: 4 * (j + 1)])
            patched_image[i][j] = 2 * (patch.flatten() / 255) - 1
    patched_tensor = torch.tensor(patched_image, dtype=torch.float32)
    return patched_tensor


class PixelDataset(Dataset):

    def __init__(self, filepath = 'Datasets/mnist_train.csv', samples=1000000):
        labels, images = load_mnist(filepath, samples=samples)
        self.labels = labels
        self.patched_images = [to_patches(im) for im in images]
        self.raw_images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patched_images[idx], self.labels[idx]

if __name__ == "__main__":
    pixel = PixelDataset('Datasets/mnist_train.csv')

    im = pixel.raw_images[0]
    show(im)
    show(2 *(im/255) - 1)