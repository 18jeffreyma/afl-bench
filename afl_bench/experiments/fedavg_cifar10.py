import logging
from typing import List

import torch

import wandb
from afl_bench.agents import ClientThread, Server, Strategy
from afl_bench.agents.clients import Client
from afl_bench.agents.runtime_model import InstantRuntime
from afl_bench.datasets.uniform.cifar10 import load_cifar10
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# Optional: set logging level to DEBUG to see more detailed logs.
logging.basicConfig(level=logging.DEBUG)

# Run parameters.

# Start a new wandb run to track this script
run = wandb.init(
    # set the wandb project where this run will be logged
    project="afl-bench",
    entity="afl-bench",
    # track hyperparameters and run metadata
    config={
        "description": "FedAvg on CIFAR-100",
        "architecture": "SimpleCNN",
        "dataset": "CIFAR10",
        "wait_for_full": True,
        "buffer_size": 4,
        "ms_to_wait": None,
        "num_clients": 4,
        "client_lr": 0.001,
        "num_server_aggregations": 10,
    },
)


def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
    """
    Define strategy and aggregation function.
    """
    # Client updates have the following nesting:
    #  - client_update[i] = (old_model, new_model, old_model_index) is the i-th update in the buffer
    # We pivot this such that we have a list of old models, new models to diff updates.
    old_models, new_models, _ = tuple(zip(*client_updates))

    # Return new model that is just the parameter wise average of the previous.
    per_pair_updates = [
        [new_param - old_param for old_param, new_param in zip(old_model, new_model)]
        for old_model, new_model in zip(old_models, new_models)
    ]

    # For each corresponding parameter group we compute the average of the updates
    updates = [
        torch.mean(torch.stack(param_group, 0), 0)
        for param_group in zip(*per_pair_updates)
    ]

    # assert [t.shape for t in global_model] == [t.shape for t in new_model]
    return [
        global_param.add(update) for global_param, update in zip(global_model, updates)
    ]


# Note that we wait for full buffer and specify buffer size.
strategy = Strategy(
    name="JeffTest",
    wait_for_full=run.config["wait_for_full"],
    buffer_size=run.config["buffer_size"],
    ms_to_wait=run.config["ms_to_wait"],
    aggregate=aggregation_func,
)

# Define a server where global model is a SimpleCNN and strategy is the one defined above.
server = Server(CIFAR10SimpleCNN(), strategy, run.config["num_server_aggregations"])

# Assemble a list of all client threads.
trainloaders, testloaders, _ = load_cifar10(run.config["num_clients"])

# Create client threads with models. Note runtime is instant (meaning no simulated training delay).
client_threads = []
for i in range(run.config["num_clients"]):
    client = Client(
        CIFAR10SimpleCNN(), trainloaders[i], testloaders[i], run.config["client_lr"]
    )
    client_thread = ClientThread(
        client, server, runtime_model=InstantRuntime(), client_id=i
    )
    client_threads.append(client_thread)

# Start up server and start up all client threads.
server.run()
for client_thread in client_threads:
    client_thread.run()

# Kill client threads once server stops.
server.join()
for client_thread in client_threads:
    client_thread.stop()

wandb.finish()
