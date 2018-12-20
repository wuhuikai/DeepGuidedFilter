import torch
import torch.nn as nn

from guided_filter import FastGuidedFilter


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


def build_lr_net(norm=AdaptiveNorm, layer=5):
    layers = [
        nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    for l in range(1, layer):
        layers += [nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
                   norm(24),
                   nn.LeakyReLU(0.2, inplace=True)]

    layers += [
        nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(24),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(24,  3, kernel_size=1, stride=1, padding=0, dilation=1)
    ]

    net = nn.Sequential(*layers)

    return net


class DeepGuidedFilter(torch.jit.ScriptModule):
    def __init__(self):
        super(DeepGuidedFilter, self).__init__()
        self.lr = torch.jit.trace(build_lr_net(), torch.randn(1, 3, 64, 64))
        self.gf = FastGuidedFilter()

    @torch.jit.script_method
    def forward(self, x_lr:torch.Tensor, radius:int=1, eps:float=1e-8):
        return self.gf(x_lr, self.lr(x_lr), radius, eps)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))
