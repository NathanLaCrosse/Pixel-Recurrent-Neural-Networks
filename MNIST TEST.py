import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

mnist = datasets.MNIST(root='./data', train=False, download=True, transform=None)