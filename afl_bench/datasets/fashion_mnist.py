import os
from functools import cache
from typing import List, Tuple

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from afl_bench.datasets import RAW_DATA_PATH
from afl_bench.datasets.common import (
    load_datasets_iid,
    load_datasets_one_class_per_client,
    load_datasets_sorted_partition,
)


def get_fashion_mnist():
    """
    Download or load FashionMNIST dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    path = os.path.join(RAW_DATA_PATH, "fashion_mnist")
    train_set = FashionMNIST(path, train=True, download=True, transform=transform)
    test_set = FashionMNIST(path, train=False, download=True, transform=transform)
    return train_set, test_set


@cache
def load_fashion_mnist_iid(num_clients: int, batch_size=32):
    """
    Split FashionMNIST train set into train/eval sets and create DataLoaders.

    Args:
        num_clients (int): Number of dataloaders to return.
        batch_size (int, optional): Batch size for each dataloader. Defaults to 32.

    Returns:
        tuple: (train loaders, eval loaders, test loader)
    """
    train_set, test_set = get_fashion_mnist()
    return load_datasets_iid(train_set, test_set, num_clients, batch_size=batch_size)


@cache
def load_fashion_mnist_sorted_partition(
    num_clients, batch_size=32
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    train_set, test_set = get_fashion_mnist()

    return load_datasets_sorted_partition(
        train_set, test_set, num_clients, batch_size=batch_size
    )


@cache
def load_fashion_mnist_one_class_per_client(
    num_clients, batch_size=32
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Return dataloader such that each client has one class.
    """
    assert num_clients % 10 == 0

    train_set, test_set = get_fashion_mnist()

    return load_datasets_one_class_per_client(
        train_set, test_set, 10, num_clients, batch_size=batch_size
    )
