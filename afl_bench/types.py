from typing import List, Tuple, TypeAlias

import torch

ModelParams: TypeAlias = List[torch.Tensor]
ClientUpdate: TypeAlias = Tuple[ModelParams, ModelParams, int]
