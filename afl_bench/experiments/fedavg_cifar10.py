import logging
from typing import List

import torch

import wandb
from afl_bench.agents import ClientThread, Server, Strategy
from afl_bench.agents.clients import Client
from afl_bench.agents.runtime_model import InstantRuntime
from afl_bench.datasets.cifar10 import (
    load_cifar10_iid,
    load_cifar10_one_class_per_client,
    load_cifar10_sorted_partition,
)
from afl_bench.experiments.utils import get_cmd_line_parser
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# Set logging level to DEBUG to see more detailed logs.
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Run parameters.
args = get_cmd_line_parser()

if __name__ == "__main__":
    # Start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="afl-bench",
        entity="afl-bench",
        # track hyperparameters and run metadata
        config={
            "name": f"FedAvg CIFAR-10 {args['data_distribution']}, {args['num_clients']} clients",
            "description": "FedAvg on CIFAR-10",
            "architecture": "SimpleCNN",
            "dataset": "CIFAR10",
            "data_distribution": args["data_distribution"],
            "wait_for_full": args["wait_for_full"],
            "buffer_size": args["buffer_size"],
            "ms_to_wait": args["ms_to_wait"],
            "num_clients": args["num_clients"],
            "client_lr": args["client_lr"],
            "num_aggregations": args["num_aggregations"],
            "batch_size": args["batch_size"],
            "device": "cuda",
        },
    )

    def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
        """
        Define strategy and aggregation function.
        """
        # Client updates have the following nesting:
        #  - client_update[i] = (old_model, new_model, old_model_index) is the i-th update in
        #    the buffer
        # We pivot this such that we have a list of old models, new models to diff updates.
        old_models, new_models, _ = tuple(zip(*client_updates))

        # Return new model that is just the parameter wise average of the previous.
        per_pair_updates = [
            [
                new_param - old_param
                for old_param, new_param in zip(old_model, new_model)
            ]
            for old_model, new_model in zip(old_models, new_models)
        ]

        # For each corresponding parameter group we compute the average of the updates
        updates = [
            torch.mean(torch.stack(param_group, 0), 0)
            for param_group in zip(*per_pair_updates)
        ]

        # assert [t.shape for t in global_model] == [t.shape for t in new_model]
        return [
            global_param.add(update)
            for global_param, update in zip(global_model, updates)
        ]

    # Note that we wait for full buffer and specify buffer size.
    strategy = Strategy(
        name="JeffTest",
        wait_for_full=run.config["wait_for_full"],
        buffer_size=run.config["buffer_size"],
        ms_to_wait=run.config["ms_to_wait"],
        aggregate=aggregation_func,
    )

    # Choose dataset distribution load based on run config.
    load_functions = {
        "iid": load_cifar10_iid,
        "one_class_per_client": load_cifar10_one_class_per_client,
        "sorted_partition": load_cifar10_sorted_partition,
    }
    load_function = load_functions[run.config["data_distribution"]]

    trainloaders, testloaders, global_testloader = load_function(
        run.config["num_clients"], batch_size=run.config["batch_size"]
    )

    # Instantiate runtime models.
    runtime_models = [InstantRuntime() for _ in range(run.config["num_clients"])]

    #########################################################################################
    # NOTE: NOTHING BELOW THIS LINE SHOULD BE CHANGED.                                      #
    #########################################################################################

    # Define a server where global model is a SimpleCNN and strategy is the one defined above.
    server = Server(
        CIFAR10SimpleCNN().to(run.config["device"]),
        strategy,
        run.config["num_aggregations"],
        global_testloader,
        device=run.config["device"],
    )

    # Create client threads with models. Note runtime is instant
    # (meaning no simulated training delay).
    client_threads = []
    for i in range(run.config["num_clients"]):
        client = Client(
            CIFAR10SimpleCNN().to(run.config["device"]),
            trainloaders[i],
            testloaders[i],
            run.config["client_lr"],
            device=run.config["device"],
        )
        client_thread = ClientThread(
            client, server, runtime_model=runtime_models[i], client_id=i
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
