import logging
from functools import partial
from typing import List, Tuple

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


# Define Exponential Weighting strategy.
def aggregation_func(
    global_model_and_version: Tuple[ModelParams, int],
    client_updates: List[ClientUpdate],
):
    """
    Define strategy and aggregation function.
    """
    global_model, version = global_model_and_version

    # Get list of client models.
    new_models: List[ModelParams] = tuple(zip(*client_updates))[1]
    prev_model_versions: List[int] = tuple(zip(*client_updates))[2]

    # Get list of length num clients with each element being a tuple of name and parameter.
    new_global_model = []
    for param_names_and_tensors, (global_param_name, global_param) in zip(
        zip(*new_models), global_model
    ):
        param_name = param_names_and_tensors[0][0]

        # Sanity check names.
        assert param_name == global_param_name
        tensors = [t for _, t in param_names_and_tensors]

        weighted_tensors = [
            t * (0.5 ** ((v - version) / 2))
            for t, v in zip(tensors, prev_model_versions)
        ]

        # Sanity check sizes.
        assert param_names_and_tensors[0][1].shape == global_param.shape

        new_param = torch.mean(torch.stack(weighted_tensors, 0), 0)

        new_global_model.append(
            (
                param_name,
                new_param if "bn" not in param_name else global_param,
            )
        )

    # Sanity check sizes.
    for a, b in zip(global_model, new_global_model):
        assert a[0] == b[0]
        assert a[1].shape == b[1].shape

    return new_global_model


# Note that we wait for full buffer and specify buffer size.
strategy = Strategy(
    name="ExpWeighting",
    wait_for_full=args["wait_for_full"],
    buffer_size=args["buffer_size"],
    ms_to_wait=args["ms_to_wait"],
    aggregate=aggregation_func,
)


if __name__ == "__main__":
    run_experiment(
        strategy=strategy, args=args, model_info=("SimpleCNN", CIFAR10SimpleCNN)
    )
