from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel

from afl_bench.types import ClientUpdate, ModelParams


class Strategy(BaseModel):
    """
    Class representing an aggregation strategy for federated learning.
    """

    # Name of strategy.
    name: str
    # Whether aggregation buffer should wait until full to perform aggregation.
    wait_for_full: bool
    # Size of aggregation buffer.
    buffer_size: Optional[int] = None
    # Whether to wait for buffer to be full or wait a number of ms for buffer.
    ms_to_wait: Optional[int] = None
    # Aggregation function with following args in order, returning a new set of model params:
    # - List of parameters for current global model to be updated in place.
    # - List of tuples of three elements (where each element is communicated update from a client):
    #   - List of parameters for initial old model of client.
    #   - List of parameters for updated new model of client after local data
    #   - Version number of global model used by client during training.
    aggregate: Callable[[Tuple[ModelParams, int], List[ClientUpdate]], ModelParams]
