import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

MEAN = 0.5
STD = 0.5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])

train_set = datasets.mnist.FashionMNIST('./data', train=True, transform=transform, download=True)
test_set = datasets.mnist.FashionMNIST('./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=4, num_workers=2)
test_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=4, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def matplotlib_imshow(img, one_chanel=False):
    if one_chanel:
        img = img.mean(dim=0)
    img = img * STD + MEAN
    npimg = img.numpy()
    if one_chanel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x
