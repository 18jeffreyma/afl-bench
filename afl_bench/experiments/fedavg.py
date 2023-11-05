from typing import List
import logging

import torch

from afl_bench.agents import Server, ClientThread, Strategy
from afl_bench.agents.clients import Client
from afl_bench.agents.runtime_model import InstantRuntime
from afl_bench.datasets.uniform.cifar10 import load_datasets
from afl_bench.models import SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# set logger level
logging.basicConfig(level=logging.DEBUG)


# Define strategy and aggregation function.
def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
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


strategy = Strategy(
    name="JeffTest", wait_for_full=True, buffer_size=4, aggregate=aggregation_func
)

server = Server(SimpleCNN(), strategy)

# Assemble a list of all client threads.
num_clients = 4
trainloaders, testloaders, _ = load_datasets(num_clients)

client_threads = []
for i in range(num_clients):
    client = Client(SimpleCNN(), trainloaders[i], testloaders[i])
    client_thread = ClientThread(client, server, runtime_model=InstantRuntime())
    client_threads.append(client_thread)

# Start up server and start up all client threads.
server.run()
for client_thread in client_threads:
    client_thread.run()
