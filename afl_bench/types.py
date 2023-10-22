from typing import List, Tuple, TypeAlias
from torch.nn.parameter import Parameter

ModelParams: TypeAlias = List[Parameter]
ClientUpdate: TypeAlias = Tuple[ModelParams, ModelParams, int]