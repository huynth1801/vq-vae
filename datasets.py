import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def prepare_dataset(batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    train_dataset = datasets.CIFAR10(root='data',
                           train=True,
                           transform=transform,
                           download=True)

    valid_dataset = datasets.CIFAR10(root='data',
                                    train=False,
                                    transform=transform,
                                     download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader

if __name__=='__main__':
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    _ ,_ , train_dataloader, valid_dataloader = prepare_dataset(128)
    real_batch = next(iter(train_dataloader))
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title("Example Data")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                                              padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()