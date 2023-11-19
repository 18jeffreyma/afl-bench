import torch
from torch.profiler import ProfilerActivity, profile, record_function

from afl_bench.agents.common import get_parameters, set_parameters


class Client:
    def __init__(
        self, net, trainloader, valloader, num_steps=10, lr=0.001, device="cpu"
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_steps = num_steps
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        avg_epoch_loss, avg_epoch_acc = _train(
            self.net,
            self.trainloader,
            self.optimizer,
            self.num_steps,
            device=self.device,
            lr=self.lr,
        )
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
        loss, accuracy = _test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def _train(net, trainloader, optimizer, num_steps: int, device="cpu", lr=0.001):
    """Train the network on the training set."""

    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    net.train()

    step_count = 0
    total_count = 0
    correct_count = 0

    keep_running = True

    while keep_running:
        for images, labels in trainloader:
            if step_count >= num_steps:
                keep_running = False
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                total_count += labels.size(0)
                correct_count += (torch.argmax(outputs.data, 1) == labels).sum().item()

            step_count += 1

    return loss.detach().clone(), correct_count / total_count


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
            total += labels.size(0)
            predicted = torch.argmax(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
