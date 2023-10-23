import torch

from typing import List, Tuple, TypeAlias


ModelParams: TypeAlias = List[torch.Tensor]
ClientUpdate: TypeAlias = Tuple[ModelParams, ModelParams, int]