from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from afl_bench.datasets.utils import one_class_partition, sort_and_partition


def load_datasets_iid(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_clients: int,
    batch_size=32,
    pin_memory=True,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Split training set into 10 partitions to simulate the individual dataset.
    """
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(
            DataLoader(
                ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
            )
        )
        valloaders.append(
            DataLoader(ds_val, batch_size=batch_size, pin_memory=pin_memory)
        )
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory)
    return trainloaders, valloaders, testloader


def load_datasets_sorted_partition(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_clients: int,
    batch_size=32,
    pin_memory=True,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Load datasets such that each client has a sorted even-length partition of the dataset.
    """

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients

    datasets = sort_and_partition(train_set, train_set.targets, lengths)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(
            DataLoader(
                ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
            )
        )
        valloaders.append(
            DataLoader(ds_val, batch_size=batch_size, pin_memory=pin_memory)
        )

    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory)
    return trainloaders, valloaders, testloader


def load_datasets_one_class_per_client(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_classes: int,
    num_clients: int,
    batch_size=32,
    pin_memory=True,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Return dataloader such that each client has only one class.
    """
    assert num_clients % num_classes == 0

    datasets = one_class_partition(
        train_set, train_set.targets, num_classes, num_clients
    )

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(
            DataLoader(
                ds_train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
            )
        )
        valloaders.append(
            DataLoader(ds_val, batch_size=batch_size, pin_memory=pin_memory)
        )

    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory)
    return trainloaders, valloaders, testloader