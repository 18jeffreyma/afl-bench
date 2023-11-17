import logging
import random
from functools import partial
from typing import List, Tuple

import numpy as np
import torch

import wandb
from afl_bench.agents import Strategy
from afl_bench.experiments.utils import get_cmd_line_parser, run_experiment
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN
from afl_bench.types import ClientUpdate, ModelParams

# Set random seed for reproducibility.
SEED = 42
random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)

# Set logging level to DEBUG to see more detailed logs.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Run parameters.
args = get_cmd_line_parser()
num_clients = len(args["client_runtimes"])
buffer_size = args["buffer_size"]


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
    old_models, new_models, prev_model_versions = tuple(zip(*client_updates))

    # Compute weights for each client update, weighting more recent updates more heavily.
    logger.info("version: %s", version)
    logger.info("staleness: %s", version - np.array(prev_model_versions))
    wandb.log({"average_staleness": np.mean(version - np.array(prev_model_versions))})
    weights = [
        (1 + (version - v) * buffer_size) / num_clients * 0.1
        for v in prev_model_versions
    ]
    logger.info("Aggregation weights: %s", weights)

    # Get list of length num clients with each element being a tuple of name and parameter.
    new_global_model = []
    for (
        param_names_and_tensors,
        old_param_names_and_tensors,
        (global_param_name, global_param),
    ) in zip(zip(*new_models), zip(*old_models), global_model):
        param_name = param_names_and_tensors[0][0]

        # Sanity check names.
        assert param_name == global_param_name
        assert len(weights) == len(param_names_and_tensors)
        raw_updates = [
            t - old_t
            for (_, t), (_, old_t) in zip(
                param_names_and_tensors, old_param_names_and_tensors
            )
        ]
        # Normalize each update by its norm.
        normalized_update = [raw_update for raw_update in raw_updates]
        weighted_updates = [w * t for w, t in zip(weights, normalized_update)]

        # Sanity check sizes.
        assert param_names_and_tensors[0][1].shape == global_param.shape

        # Compute new param as weighted average of updates.
        computed_update = torch.sum(torch.stack(weighted_updates, 0), 0)
        new_param = global_param + computed_update

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
    name="FedFair",
    wait_for_full=args["wait_for_full"],
    buffer_size=args["buffer_size"],
    ms_to_wait=args["ms_to_wait"],
    aggregate=aggregation_func,
)


if __name__ == "__main__":
    run_experiment(
        strategy=strategy, args=args, model_info=("SimpleCNN", CIFAR10SimpleCNN)
    )
