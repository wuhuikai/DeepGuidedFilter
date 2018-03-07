import torch
import torch.nn as nn

from torch.nn import init

from guided_filter.guided_filter import FastGuidedFilter

def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        init.constant(m.bias.data,   0.0)

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

    net.apply(weights_init_identity)

    return net

class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=1, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        self.lr = build_lr_net()
        self.gf = FastGuidedFilter(radius, eps)

    def forward(self, x_lr, x_hr):
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    def __init__(self, radius=1, eps=1e-4):
        super(DeepGuidedFilterAdvanced, self).__init__(radius, eps)

        self.guided_map = nn.Sequential(
            nn.Conv2d(3, 15, 1, bias=False),
            AdaptiveNorm(15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(15, 3, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        # return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr)).clamp(0, 1)
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))

from torch.nn import functional as F

class DeepGuidedFilterConv(nn.Module):
    def __init__(self):
        super(DeepGuidedFilterConv, self).__init__()
        self.lr = build_lr_net()
        self.gf = nn.Sequential(
            nn.Conv2d(6,  24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            AdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            AdaptiveNorm(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24,  3, kernel_size=1, stride=1, padding=0, dilation=1)
        )

        self.gf.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        y_lr = self.lr(x_lr)
        y_hr = F.upsample(y_lr, x_hr.size()[-2:], mode='bilinear')

        return self.gf(torch.cat((y_hr, x_hr), dim=1).detach()).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))

class DeepGuidedFilterDJF(nn.Module):
    def __init__(self):
        super(DeepGuidedFilterDJF, self).__init__()
        self.lr = build_lr_net()
        self.stream_x = nn.Sequential(
            nn.Conv2d(3,  96, kernel_size=9, stride=1, padding=4, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 1,  kernel_size=5, stride=1, padding=2, dilation=1)
        )

        self.stream_y = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=9, stride=1, padding=4, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 1, kernel_size=5, stride=1, padding=2, dilation=1)
        )

        self.combined = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2, dilation=1)
        )

        self.stream_x.apply(weights_init_identity)
        self.stream_y.apply(weights_init_identity)
        self.combined.apply(weights_init_identity)

    def forward(self, x_lr, x_hr):
        y_lr = self.lr(x_lr)
        y_hr = F.upsample(y_lr, x_hr.size()[-2:], mode='bilinear')

        x_hr = self.stream_x(x_hr)
        y_hr = self.stream_y(y_hr)

        return self.combined(torch.cat((x_hr, y_hr), dim=1)).clamp(0, 1)

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))