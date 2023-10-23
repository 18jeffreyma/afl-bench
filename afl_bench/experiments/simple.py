
from typing import List
from afl_bench.agents import Server, ClientThread, RuntimeModel, Strategy
from afl_bench.agents.runtime_model import InstantRuntime
from afl_bench.clients.simple import SimpleClient
from afl_bench.datasets.uniform.cifar10 import load_datasets
from afl_bench.types import ClientUpdate, ModelParams

from afl_bench.models import SimpleCNN

import numpy as np

import logging
# set logger level
logging.basicConfig(level=logging.DEBUG)

# Define strategy and aggregation function.
def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
    # Return new model that is just the parameter wise average of the previous.
    new_model = []
    
    # Iterate over all corresponding new params and take corresponding averages.
    for param_group in zip(*[new_param for _, new_param, _ in client_updates]):
        new_model.append(np.mean(param_group, axis=0))
    
    assert [tuple(t.shape) for t in global_model] == [t.shape for t in new_model]
    return new_model

strategy = Strategy(
    name="JeffTest",
    wait_for_full=True,
    buffer_size=4,
    aggregate=aggregation_func)

server = Server(SimpleCNN(), strategy)

# Assemble a list of all client threads.
num_clients = 10
trainloaders, testloaders, _ = load_datasets(num_clients)

client_threads = []
for i in range(num_clients):
    client = SimpleClient(SimpleCNN(), trainloaders[i], testloaders[i])
    client_thread = ClientThread(client, server, runtime_model=InstantRuntime())
    client_threads.append(client_thread)
    
# Start up server and start up all client threads.
server.run()
for client_thread in client_threads:
    client_thread.run()