import logging
from functools import partial
from typing import List

import torch

import wandb
from afl_bench.agents import ClientThread, Server, Strategy
from afl_bench.agents.clients import Client
from afl_bench.experiments.utils import get_cmd_line_parser
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# Set logging level to DEBUG to see more detailed logs.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Always use CUDA if available, otherwise use MPS if available, otherwise use CPU.
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Run parameters.
args = get_cmd_line_parser()


# Define FedAvg strategy.
def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
    """
    Define strategy and aggregation function.
    """

    # Get list of client models.
    new_models: List[ModelParams] = tuple(zip(*client_updates))[1]

    # Get list of length num clients with each element being a tuple of name and parameter.
    new_global_model = []
    for param_names_and_tensors, (global_param_name, global_param) in zip(
        zip(*new_models), global_model
    ):
        param_name = param_names_and_tensors[0][0]

        # Sanity check names.
        assert param_name == global_param_name
        tensors = [t for _, t in param_names_and_tensors]

        # Sanity check sizes.
        assert param_names_and_tensors[0][1].shape == global_param.shape
        new_global_model.append(
            (
                param_name,
                torch.mean(
                    torch.stack(tensors, 0),
                    0,
                )
                if "bn" not in param_name
                else global_param,
            )
        )

    # Sanity check sizes.
    for a, b in zip(global_model, new_global_model):
        assert a[0] == b[0]
        assert a[1].shape == b[1].shape

    return new_global_model


# Note that we wait for full buffer and specify buffer size.
strategy = Strategy(
    name="FedAvg",
    wait_for_full=args["wait_for_full"],
    buffer_size=args["buffer_size"],
    ms_to_wait=args["ms_to_wait"],
    aggregate=aggregation_func,
)


if __name__ == "__main__":
    # Start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="afl-bench",
        entity="afl-bench",
        name=f"{strategy.name} {args['dataset']} {args['data_distribution'] + ((' ' + str(args['num_remove'])) if args['num_remove'] is not None else '')}, client info {args['client_info']}, buffer size {args['buffer_size']}",
        # track hyperparameters and run metadata
        config={
            "architecture": "CIFAR10SimpleCNN",
            "dataset": args["dataset"],
            "data_distribution": args["data_distribution"],
            "num_remove": args["num_remove"],
            "wait_for_full": args["wait_for_full"],
            "buffer_size": args["buffer_size"],
            "ms_to_wait": args["ms_to_wait"],
            "num_clients": len(args["client_runtimes"]),
            "client_runtimes": args["client_runtimes"],
            "client_lr": args["client_lr"],
            "client_num_steps": args["client_num_steps"],
            "num_aggregations": args["num_aggregations"],
            "batch_size": args["batch_size"],
            "device": device,
        },
    )

    trainloaders, testloaders, global_testloader = args["load_function"](
        run.config["num_clients"], batch_size=run.config["batch_size"]
    )

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
    for i, runtime_model in enumerate(args["client_runtimes"]):
        client = Client(
            CIFAR10SimpleCNN().to(run.config["device"]),
            trainloaders[i],
            testloaders[i],
            num_steps=run.config["client_num_steps"],
            lr=run.config["client_lr"],
            device=run.config["device"],
        )
        client_thread = ClientThread(
            client, server, runtime_model=runtime_model, client_id=i
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
