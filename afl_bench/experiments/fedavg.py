import logging
from functools import partial
from typing import List

import torch

import wandb
from afl_bench.agents import ClientThread, Server, Strategy
from afl_bench.agents.clients import Client
from afl_bench.experiments.utils import get_cmd_line_parser, run_experiment
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# Set logging level to DEBUG to see more detailed logs.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    run_experiment(
        strategy=strategy, args=args, model_info=("SimpleCNN", CIFAR10SimpleCNN)
    )
