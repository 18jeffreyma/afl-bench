from typing import List

import torch


def get_parameters(net) -> List[torch.Tensor]:
    return [val for val in net.parameters()]


def set_parameters(net, parameters: List[torch.Tensor]):
    for p, new_p in zip(net.parameters(), parameters):
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()
        # Update the parameter.
        p.data.copy_(new_p)
