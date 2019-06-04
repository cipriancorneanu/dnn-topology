'''
Data loading Utilities for preparing for various different datasets.
Includes, MNIST, CIFAR10, TinyImagenet.
For MNIST and CIFAR10 there are special adversarial samples prepared
for evaluation. -> <dataset>_adversarial(). Each function returns a
train and a test DataLoader except the dedicated functions for adversarial
samples that return a single loader.
'''

import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import *
import random
import numpy as np


TRANSFORMS_TR = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_COLOR32 = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x : x.view(1, 32, 32).expand(3, -1, -1)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_COLOR32 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x : x.view(1, 32, 32).expand(3, -1, -1)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_CIFAR10_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_SVHN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_SVHN_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_SVHN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_SVHN_GRAY28 = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))       
])

TRANSFORMS_TE_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRANSFORMS_MNIST_ADV = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

TRANSFORMS_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])


def loader(data, batch_size, subset=[], sampling=-1):
    ''' Interface to the dataloader function '''
    if data == 'mnist_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_MNIST, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_MNIST, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    if data == 'mnist_color32_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_TR_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'mnist_color32_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_TE_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'cifar10_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'cifar10_gray28_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'cifar10_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling , num_workers=2, subset=subset)
    elif data == 'cifar10_gray28_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'svhn_train':
        return dataloader('svhn', './data', train='train', transform=TRANSFORMS_TR_SVHN, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'svhn_test':
        return dataloader('svhn', './data', train='test', transform=TRANSFORMS_TE_SVHN, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'svhn_gray28_train':
        return dataloader('svhn', './data', train='train', transform=TRANSFORMS_TR_SVHN_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'svhn_gray28_test':
        return dataloader('svhn', './data', train='test', transform=TRANSFORMS_TE_SVHN_GRAY28, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'fashion_mnist_train':
        return dataloader('fashion_mnist', './data', train=True, transform=TRANSFORMS_TR, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'fashion_mnist_test':
        return dataloader('fashion_mnist', './data', train=False, transform=TRANSFORMS_TE, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'fashion_mnist_color32_train':
        return dataloader('fashion_mnist', './data', train=True, transform=TRANSFORMS_TR_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'fashion_mnist_color32_test':
        return dataloader('fashion_mnist', './data', train=False, transform=TRANSFORMS_TE_COLOR32, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'lenet_mnist_adversarial_test':
        return dataloader('/data/data1/datasets/lenet_mnist_adversarial/', train=False,
                          transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'lenet_cifar10_adversarial_test':
        return dataloader('/data/data1/datasets/lenet_cifar_adversarial/', train=False,
                          transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'vgg_cifar10_adversarial_test':
        return dataloader('vgg_cifar10_adversarial', '/data/data1/datasets/vgg_cifar_adversarial/', train=False, transform=TRANSFORMS_TE_CIFAR10, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'imagenet_train':
        return dataloader('tinyimagenet', '/data/data1/datasets/tiny-imagenet-200/train/',
                                 train=True, transform=TRANSFORMS_TR_IMAGENET, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)
    elif data == 'imagenet_test':
        return dataloader('tinyimagenet', '/data/data1/datasets/tiny-imagenet-200/val/images/',
                                 train=False, transform=TRANSFORMS_TE_IMAGENET, batch_size=batch_size, sampling=sampling, num_workers=2, subset=subset)

    
def get_dataset(data, path, train, transform):
    ''' Return loader for torchvision data. If data in [mnist, cifar] torchvision.datasets has built-in loaders else load from ImageFolder '''
    if data == 'mnist':
        dataset = torchvision.datasets.MNIST(path, train=train, download=True, transform=transform)
    elif data == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transform)
    elif data == 'svhn':
        dataset = torchvision.datasets.SVHN(path, split=train, download=True, transform=transform)
    elif data == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(path, train=train, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    return dataset


def dataloader(data, path, train, transform, batch_size, num_workers, subset=[], sampling=-1):
    dataset = get_dataset(data, path, train, transform)
        
    if subset:
        dataset = torch.utils.data.Subset(dataset, subset)

    if sampling == -1:
        sampler = SequentialSampler(dataset)
    elif sampling == -2:
        sampler = RandomSampler(dataset)
    else:
        sampler = BinarySampler(dataset, sampling)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)


class BinarySampler(Sampler):
    """One-vs-rest sampling where pivot indicates the target class """

    def __init__(self, dataset, pivot):
        self.dataset = dataset
        self.pivot_indices = self._get_pivot_indices(pivot)
        self.nonpivot_indices = self._get_nonpivot_indices()
        self.indices = self._get_indices()
    
    def _get_targets(self):
        return [x for (_, x) in self.dataset]

    def _get_pivot_indices(self, pivot):
        return [i for i, x in enumerate(self._get_targets()) if x==pivot]

    def _get_nonpivot_indices(self):
        return random.sample(list(set(np.arange(len(self.dataset)))-set(self.pivot_indices)), len(self.pivot_indices))

    def _get_indices(self):
        return self.pivot_indices + self.nonpivot_indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
        
    def __len__(self):
        return len(self.indices)
