from functools import cache
import os
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from afl_bench.datasets import RAW_DATA_PATH


@cache
def load_datasets(
    num_clients: int, batch_size=32
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    path = os.path.join(RAW_DATA_PATH, "cifar10")
    trainset = CIFAR10(path, train=True, download=True, transform=transform)
    testset = CIFAR10(path, train=False, download=True, transform=transform)

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
