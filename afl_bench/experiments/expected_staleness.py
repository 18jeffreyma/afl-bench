import logging
import random
from collections import defaultdict
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
num_clients = len(args["client_runtimes"])
buffer_size = args["buffer_size"]


class StalenessTracker:
    def __init__(self, window_size=5) -> None:
        self.window_size = window_size
        self.client_update = defaultdict(list)

    def track_update(self, client_id: int, staleness: int):
        # Pop oldest update if window size is reached.
        if len(self.client_update[client_id]) >= self.window_size:
            self.client_update[client_id].pop()
        self.client_update[client_id].append(staleness)

    def get_avg_staleness(self, client_id: int):
        if len(self.client_update[client_id]) == 0:
            return 0.0
        return sum(self.client_update[client_id]) / len(self.client_update[client_id])


staleness_tracker = StalenessTracker()


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
    client_ids, old_models, new_models, prev_model_versions = tuple(
        zip(*client_updates)
    )

    # Compute weights for each client update, weighting more recent updates more heavily.
    for client_id, v in zip(client_ids, prev_model_versions):
        staleness_tracker.track_update(client_id, version - v)

    stalenesses = [
        staleness_tracker.get_avg_staleness(client_id) for client_id in client_ids
    ]

    weights = [s * buffer_size / num_clients for s in stalenesses]
    weight_total = sum(weights)
    weights = [weight / weight_total if weight_total > 0 else 0 for weight in weights]

    logger.info("Aggregation clients: %s", client_ids)
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
    name="ExpectedStaleness",
    wait_for_full=args["wait_for_full"],
    buffer_size=args["buffer_size"],
    ms_to_wait=args["ms_to_wait"],
    aggregate=aggregation_func,
)


if __name__ == "__main__":
    run_experiment(strategy=strategy, args=args, model_info=args["model_info"])
