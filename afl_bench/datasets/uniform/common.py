from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split


def load_datasets(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_clients: int,
    batch_size=32,
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
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(test_set, batch_size=batch_size)
    return trainloaders, valloaders, testloader
