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
def load_mnist(filepath):
    mnist_train = pd.read_csv(filepath)
    labels = []
    images = []
    for _, row in tqdm.tqdm(mnist_train.iterrows()):
        pict = np.zeros((28, 28), dtype=np.float32)
        for j in range(28):
            for k in range(28):
                pict[j, k] = row[f'{j + 1}x{k + 1}']
        labels.append(row['label'])
        images.append(pict)
        break
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
    patched_tensor = torch.tensor(patched_image)
    return patched_tensor


class PixelDataset(Dataset):

    def __init__(self, filepath = 'Datasets/mnist_train.csv'):
        labels, images = load_mnist(filepath)
        self.labels = labels
        self.patched_images = [to_patches(im) for im in images]
        self.raw_images = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patched_images[idx], self.labels[idx]

pixel = PixelDataset('Datasets/mnist_train.csv')

im = pixel.raw_images[0]
show(im)
show(2 *(im/255) - 1)