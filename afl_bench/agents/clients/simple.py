import torch

from afl_bench.agents.common import get_parameters, set_parameters


class Client:
    def __init__(self, net, trainloader, valloader, lr=0.001):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.lr = lr

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        avg_epoch_loss, avg_epoch_acc = _train(self.net, self.trainloader, epochs=1)
        return (
            get_parameters(self.net),
            len(self.trainloader),
            {
                "avg_loss": float(avg_epoch_loss),
                "avg_accuracy": float(avg_epoch_acc),
                # TODO(jeff): add more metrics here?
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = _test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def _train(net, trainloader, epochs: int, device="cpu", verbose=False, lr=0.001):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    total_epoch_losses = 0.0
    total_epoch_accs = 0.0

    for _ in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            with torch.no_grad():
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        with torch.no_grad():
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total

            total_epoch_losses += epoch_loss
            total_epoch_accs += epoch_acc

    return total_epoch_losses / epochs, total_epoch_accs / epochs


def _test(net, testloader, device="cpu"):
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
