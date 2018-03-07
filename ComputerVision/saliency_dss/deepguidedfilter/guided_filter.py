import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .box_filter import BoxFilter


class FastGuidedFilterColor(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilterColor, self).__init__()

        self.r = r
        self.boxfilter = BoxFilter(r)
        self.eps = eps

    def forward(self, lr_x, lr_y, hr_x):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and c_lrx == 3
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(lr_x) / N
        # mean_y
        mean_y = self.boxfilter(lr_y) / N

        # var_x
        # The variance in each local patch is a 3 x3 symmetric matrix:
        #         rr, rg, rb
        # var_x = rg, gg, gb + eps * I
        #         rb, gb, bb
        var_x = self.boxfilter(lr_x[:, [0, 0, 0, 1, 1, 2]] * lr_x[:, [0, 1, 2, 1, 2, 2]]) / N - \
                mean_x[:, [0, 0, 0, 1, 1, 2]] * mean_x[:, [0, 1, 2, 1, 2, 2]] + self.eps
        # inv_x
        inv_x = var_x[:, [3, 4, 1, 0, 2, 0]] * var_x[:, [5, 2, 4, 5, 1, 3]] - \
                var_x[:, [4, 1, 3, 2, 0, 1]] * var_x[:, [4, 5, 2, 2, 4, 1]]
        inv_x = inv_x / (inv_x[:, 0:3] * var_x[:, 0:3]).sum(1, keepdim=True)

        result = []
        for i in range(c_lry):
            # cov_xy
            cov_xy = self.boxfilter(lr_x * lr_y[:, i:i + 1]) / N - mean_x * mean_y[:, i:i + 1]

            # A
            A = inv_x[:, 0:3] * cov_xy[:, 0:1] + inv_x[:, [1, 3, 4]] * cov_xy[:, 1:2] + \
                inv_x[:, [2, 4, 5]] * cov_xy[:, 2:]
            # b
            b = mean_y[:, i:i + 1] - (A * mean_x).sum(1, keepdim=True)

            # mean_A; mean_b
            # mean_A = F.upsample(self.boxfilter(A) / N, (h_hrx, w_hrx), mode='bilinear')
            # mean_b = F.upsample(self.boxfilter(b) / N, (h_hrx, w_hrx), mode='bilinear')
            mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
            mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

            result.append((mean_A * hr_x).sum(1, keepdim=True) + mean_b)

        return torch.cat(result, dim=1)


class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter, self).__init__()

        self.r = r
        self.boxfilter = BoxFilter(r)
        self.eps = eps

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y

        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b
