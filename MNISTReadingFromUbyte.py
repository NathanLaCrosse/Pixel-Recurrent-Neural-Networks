import matplotlib.pyplot as plt
from torchvision import datasets

mnist = datasets.MNIST(root='./data', train=False, download=True, transform=None)

import gzip
f = gzip.open('data/MNIST/raw/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

import numpy as np
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

image = np.asarray(data[2]).squeeze()
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()
