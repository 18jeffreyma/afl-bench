import logging
import random
from typing import List, Tuple

import numpy as np
import torch

from afl_bench.agents import Strategy
from afl_bench.experiments.utils import get_cmd_line_parser, run_experiment
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


# Define FedAvg strategy.
def aggregation_func(
    global_model_and_version: Tuple[ModelParams, int],
    client_updates: List[ClientUpdate],
):
    """
    Define strategy and aggregation function.
    """
    global_model, version = global_model_and_version

    # Get list of client models.
    _, old_models, new_models, _ = tuple(zip(*client_updates))

    # Get list of length num clients with each element being a tuple of name and parameter.
    new_global_model = []
    for (
        new_param_names_and_tensors,
        old_param_names_and_tensors,
        (global_param_name, global_param),
    ) in zip(zip(*new_models), zip(*old_models), global_model):
        param_name = new_param_names_and_tensors[0][0]

        # Sanity check names.
        assert param_name == global_param_name

        # Compute update diffs for each param for each old/new model pair.
        updates = [
            new_param - old_param
            for (_, new_param), (_, old_param) in zip(
                new_param_names_and_tensors, old_param_names_and_tensors
            )
        ]

        # Sanity check sizes.
        assert new_param_names_and_tensors[0][1].shape == global_param.shape
        new_global_model.append(
            (
                param_name,
                global_param
                + torch.mean(
                    torch.stack(updates, 0),
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
    run_experiment(strategy=strategy, args=args, model_info=args["model_info"])
