from typing import List

import torch


def get_parameters(net) -> List[torch.Tensor]:
    return [val.detach().clone() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[torch.Tensor]):
    state_dict = dict(zip(net.state_dict().keys(), parameters))
    net.load_state_dict(state_dict, strict=True)
