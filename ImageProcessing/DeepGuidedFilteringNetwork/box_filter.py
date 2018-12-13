import torch

@torch.jit.script
def diff_x(input:torch.Tensor, r:int):
    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


@torch.jit.script
def diff_y(input:torch.Tensor, r:int):
    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class BoxFilter(torch.jit.ScriptModule):
    def __init__(self):
        super(BoxFilter, self).__init__()

    @torch.jit.script_method
    def forward(self, x:torch.Tensor, r:int):
        return diff_y(diff_x(x.cumsum(dim=2), r).cumsum(dim=3), r)
