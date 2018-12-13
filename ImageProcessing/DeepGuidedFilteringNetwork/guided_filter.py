import torch

from torch.nn import functional as F

from box_filter import BoxFilter


class FastGuidedFilter(torch.jit.ScriptModule):
    def __init__(self):
        super(FastGuidedFilter, self).__init__()
        self.boxfilter = BoxFilter()

    @torch.jit.script_method
    def forward(self, lr_x:torch.Tensor, lr_y:torch.Tensor, hr_x:torch.Tensor, r:int, eps:float):
        _, _, h_lrx, w_lrx = lr_x.size()
        _, _, h_hrx, w_hrx = hr_x.size()

        ## N
        N = self.boxfilter(torch.ones(1, 1, h_lrx, w_lrx).to(lr_x), r)

        ## mean_x
        mean_x = self.boxfilter(lr_x, r) / N
        ## mean_y
        mean_y = self.boxfilter(lr_y, r) / N
        ## cov_xy
        cov_xy = self.boxfilter(lr_x * lr_y, r) / N - mean_x * mean_y
        ## var_x
        var_x = self.boxfilter(lr_x * lr_x, r) / N - mean_x * mean_x

        ## A
        A = cov_xy / (var_x + eps)
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return mean_A*hr_x+mean_b
