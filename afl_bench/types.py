from typing import List, Tuple, TypeAlias

import torch

ModelParams: TypeAlias = List[Tuple[str, torch.Tensor]]
ClientUpdate: TypeAlias = Tuple[ModelParams, ModelParams, int]
