from typing import List, Tuple, TypeAlias
from torch.nn.parameter import Parameter

import numpy as np

ModelParams: TypeAlias = List[np.ndarray]
ClientUpdate: TypeAlias = Tuple[ModelParams, ModelParams, int]