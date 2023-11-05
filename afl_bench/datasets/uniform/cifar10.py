import os

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from afl_bench.datasets import RAW_DATA_PATH
from afl_bench.datasets.uniform.common import load_datasets


def get_cifar10():
    """
    Download or load CIFAR10 dataset.
    """
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    path = os.path.join(RAW_DATA_PATH, "cifar10")
    train_set = CIFAR10(path, train=True, download=True, transform=transform)
    test_set = CIFAR10(path, train=False, download=True, transform=transform)

    return train_set, test_set


def load_cifar10(num_clients: int, batch_size=32):
    """
    Split CIFAR10 train set into train/eval sets and create DataLoaders.

    Args:
        num_clients (int): Number of dataloaders to return.
        batch_size (int, optional): Batch size for each dataloader. Defaults to 32.

    Returns:
        tuple: (train loaders, eval loaders, test loader)
    """
    train_set, test_set = get_cifar10()
    return load_datasets(train_set, test_set, num_clients, batch_size=batch_size)
