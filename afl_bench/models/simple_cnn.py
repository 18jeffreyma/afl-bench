import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten


class CIFAR10SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super(CIFAR10SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 32, 3, padding="same")
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 32)
        self.dropout1 = nn.Dropout2d(0.3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(64, 64, 3, padding="same")
        self.gn3 = nn.GroupNorm(16, 64)
        self.gn4 = nn.GroupNorm(16, 64)
        self.dropout2 = nn.Dropout2d(0.5)

        self.conv5 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv6 = nn.Conv2d(128, 128, 3, padding="same")
        self.gn5 = nn.GroupNorm(32, 128)
        self.gn6 = nn.GroupNorm(32, 128)
        self.dropout3 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.gn7 = nn.GroupNorm(32, 128)
        self.dropout4 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn1(F.relu(self.conv1(x)))
        x = self.gn2(F.relu(self.conv2(x)))

        x = self.dropout1(self.pool(x))

        x = self.gn3(F.relu(self.conv3(x)))
        x = self.gn4(F.relu(self.conv4(x)))
        x = self.dropout2(self.pool(x))

        x = self.gn5(F.relu(self.conv5(x)))
        x = self.gn6(F.relu(self.conv6(x)))
        x = self.dropout3(self.pool(x))

        x = flatten(x, start_dim=1)
        x = self.gn7(F.relu(self.fc1(x)))
        x = self.fc2(self.dropout4(x))
        return x


class FashionMNISTSimpleCNN(nn.Module):
    def __init__(self) -> None:
        super(FashionMNISTSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.conv2 = nn.Conv2d(32, 32, 3, padding="same")
        self.gn1 = nn.GroupNorm(8, 32)
        self.gn2 = nn.GroupNorm(8, 32)
        self.dropout1 = nn.Dropout2d(0.3)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(64, 64, 3, padding="same")
        self.gn3 = nn.GroupNorm(16, 64)
        self.gn4 = nn.GroupNorm(16, 64)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.gn5 = nn.GroupNorm(32, 128)
        self.dropout4 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 28x28
        x = self.gn1(F.relu(self.conv1(x)))
        x = self.gn2(F.relu(self.conv2(x)))

        x = self.dropout1(self.pool(x))

        # 14x14
        x = self.gn3(F.relu(self.conv3(x)))
        x = self.gn4(F.relu(self.conv4(x)))
        x = self.dropout2(self.pool(x))

        # 7x7
        x = flatten(x, start_dim=1)
        x = self.gn5(F.relu(self.fc1(x)))
        x = self.fc2(self.dropout4(x))
        return x
