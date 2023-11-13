from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from afl_bench.datasets.utils import (
    one_class_partition,
    randomly_remove_labels,
    restricted_subpopulation,
    sort_and_partition,
)


def distribute_datasets(
    train_eval_datasets,
    test_dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in train_eval_datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(
            DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
        )
        valloaders.append(
            DataLoader(
                ds_val,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
        )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return trainloaders, valloaders, testloader


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

    return distribute_datasets(
        datasets, test_set, batch_size=batch_size, pin_memory=pin_memory
    )


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

    return distribute_datasets(
        datasets, test_set, batch_size=batch_size, pin_memory=pin_memory
    )


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

    return distribute_datasets(
        datasets, test_set, batch_size=batch_size, pin_memory=pin_memory
    )


def load_datasets_randomly_remove(
    num_to_remove: int,
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_classes: int,
    num_clients: int,
    batch_size=32,
    pin_memory=True,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Return dataloader such that each client is missing a random subset of classes.
    """
    assert num_to_remove < num_classes

    datasets = randomly_remove_labels(
        train_set, train_set.targets, num_classes, num_to_remove, num_clients
    )

    return distribute_datasets(
        datasets, test_set, batch_size=batch_size, pin_memory=pin_memory
    )


def load_datasets_restricted_subpopulation(
    size_subpopulations: List[int],
    labels_subpopulations: List[List[int]],
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    num_classes: int,
    num_clients: int,
    batch_size=32,
    pin_memory=True,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """
    Return dataloader such that last subpopulation clients have exclusive access to a subset of classes.
    """
    assert sum(size_subpopulations) <= num_clients

    # Test that the number of subpopulation labels is less than the number of classes.
    total_sub_population_labels = sum(
        len(label_subpopulation) for label_subpopulation in labels_subpopulations
    )
    assert total_sub_population_labels <= num_classes

    # TEst that there are no duplicates.
    assert len(set().union(*labels_subpopulations)) == total_sub_population_labels

    datasets = restricted_subpopulation(
        size_subpopulations,
        labels_subpopulations,
        train_set,
        train_set.targets,
        num_classes,
        num_clients,
    )

    return distribute_datasets(
        datasets,
        test_set,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )
