from collections import OrderedDict
from typing import List
import flwr
import torch

import numpy as np

from afl_bench.agents.common import get_parameters, set_parameters

class SimpleClient(flwr.client.NumPyClient):
    def __init__(
        self, 
        net,
        trainloader,
        valloader,
        device='cpu'
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        epochs = config.get("epochs", 1)
        
        set_parameters(self.net, parameters)
        _train(self.net, self.trainloader, epochs=epochs, device=self.device)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = _test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def _train(net, trainloader, epochs: int, verbose=False, device='cpu'):
    """Simple train implementation on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def _test(net, testloader, device='cpu'):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
