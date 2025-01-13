import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import random
import datetime

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from matplotlib.pyplot import imshow, imsave

MODEL_NAME = "DCGAN"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sample_image(G, n_noise):
    '''
        save sample 100 images from the generator
    '''
    z = torch.randn(100, n_noise).to(DEVICE)
    y_hat = G(z).view(100, 28, 28) 
    result = y_hat.cpu().data.numpy()
    img = np.zeros([280, 280])
    for j in range(10):
        img[j*28:(j+1)*28] = np.concatenate([x for x in result[j*10:(j+1)*10]], axis=1)
    return img


