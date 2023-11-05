# TODO(jeff): WIP

# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Sampler, random_split
# from torchvision.datasets import CIFAR10

# import os
# from functools import cache
# from typing import List, Tuple

# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import CIFAR10

# from afl_bench.datasets import RAW_DATA_PATH


# # Define a custom Sampler for non-uniform label distribution
# class NonUniformLabelSampler(Sampler):
#     def __init__(self, data_source, class_weights):
#         super().__init__()
#         self.data_source = data_source
#         self.class_weights = class_weights

#     def __iter__(self):
#         labels = [self.data_source.targets[i] for i in range(len(self.data_source))]
#         class_probs = [self.class_weights[label] for label in labels]
#         return iter(
#             torch.multinomial(
#                 torch.tensor(class_probs), len(class_probs), replacement=True
#             ).tolist()
#         )

#     def __len__(self):
#         return len(self.data_source)


# @cache
# def load_datasets(
#     num_clients: int, batch_size=32
# ) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
#     # Download and transform CIFAR-10 (train and test)
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     path = os.path.join(RAW_DATA_PATH, "cifar10")
#     trainset = CIFAR10(path, train=True, download=True, transform=transform)
#     testset = CIFAR10(path, train=False, download=True, transform=transform)

#     # Split training set into 10 partitions to simulate the individual dataset
#     partition_size = len(trainset) // num_clients
#     lengths = [partition_size] * num_clients
#     datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

#     # Define non-uniform class weights (you can modify these weights as per your requirement)
#         # specific_class_weights = {
#         #     0: 0.1,
#         #     1: 0.1,
#         #     2: 0.2,
#         #     3: 0.2,
#         #     4: 0.1,
#         #     5: 0.1,
#         #     6: 0.1,
#         #     7: 0.1,
#         #     8: 0.1,
#         #     9: 0.1,
#         # }

#         # # Create a data loader with non-uniform label distribution
#         # sampler = NonUniformLabelSampler(cifar10_dataset, specific_class_weights)
#         # data_loader = DataLoader(cifar10_dataset, batch_size=32, sampler=sampler)


#     # Split each partition into train/val and create DataLoader
#     trainloaders = []
#     valloaders = []
#     for ds in datasets:
#         len_val = len(ds) // 10  # 10 % validation set
#         len_train = len(ds) - len_val
#         lengths = [len_train, len_val]
#         ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
#         trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
#         valloaders.append(DataLoader(ds_val, batch_size=batch_size))
#     testloader = DataLoader(testset, batch_size=batch_size)
#     return trainloaders, valloaders, testloader
