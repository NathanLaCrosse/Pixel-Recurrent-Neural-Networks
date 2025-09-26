import numpy as np
import torch
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt

  # For Testing
def show(im):
    plt.imshow(im, cmap='grey', vmin=0, vmax=255)
    plt.show()

  # MNIST Images are 28 x 28 pixels
def load_mnist():
    mnist_train = pd.read_csv('Datasets/mnist_train.csv')
    out = []
    for _, row in tqdm.tqdm(mnist_train.iterrows()):
        pict = np.zeros((28, 28), dtype=np.uint8)
        for j in range(28):
            for k in range(28):
                pict[j, k] = row[f'{j + 1}x{k + 1}']
        out.append((row['label'], pict))
        break
    return out

yup = load_mnist()
show(yup[0][1])