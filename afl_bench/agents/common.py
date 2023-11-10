from typing import List, Tuple

import torch


def get_parameters(net) -> List[torch.Tensor]:
    return [(name, val) for name, val in net.named_parameters()]


def set_parameters(net, parameters: List[Tuple[str, torch.Tensor]]):
    for p, (_, new_p) in zip(net.parameters(), parameters):
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()
        # Update the parameter.
        p.data.copy_(new_p)
