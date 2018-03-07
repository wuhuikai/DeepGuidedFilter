import random
import time

import numpy as np
import torch
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from torch.autograd import gradcheck, Variable

from guided_filter.guided_filter import FastGuidedFilterColor

## GuidedFilter
print('GuidedFilter:')
## check forward
# forward on img
rgb = img_as_float(imread('test/rgb.jpg'))
gt  = img_as_float(imread('test/gt.jpg'))
x_w, x_h = rgb.shape[:2]
w, h = x_w // 8, x_h // 8
lr_rgb = resize(rgb, (w, h), order=0, mode='reflect')
lr_gt = resize(gt, (w, h), order=0, mode='reflect')
inputs = [Variable(torch.from_numpy(i.transpose((2, 0, 1))[None]).float().cuda()) for i in [lr_rgb, lr_gt, rgb]]
f = FastGuidedFilterColor(8, 1e-3).cuda()
start_time = time.time()
r = f(*inputs)
end_time = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time-start_time))
r = r.data.cpu().numpy().squeeze().transpose(1, 2, 0)
r = np.asarray(r.clip(0, 1)*255, dtype=np.uint8)
imsave('test/r_color.jpg', r)
hr_gt = resize(lr_gt, (x_w, x_h), order=0, mode='constant')
imsave('test/4_in_1_color.jpg', np.asarray(np.vstack([
                                                 np.hstack([rgb, hr_gt]),
                                                 np.hstack([gt,  img_as_float(r)])
                                           ])*255, dtype=np.uint8))

# backward | check grad
test = gradcheck(FastGuidedFilterColor(2, random.random()).double(),
                      (Variable(torch.rand((2, 3,  6,  7)).double(), requires_grad=True),
                       Variable(torch.rand((2, 2,  6,  7)).double(), requires_grad=True),
                       Variable(torch.rand((2, 3, 11, 12)).double(), requires_grad=True)), eps=1e-6, atol=1e-4)
print('\tGrad Check Result[CPU]:', test)

inputs = torch.rand((2, 3,  6,  7)), torch.rand((2, 2,  6,  7)), torch.rand((2, 3, 11, 12))
grad_out = torch.rand((2, 2, 11, 12))

fgf = FastGuidedFilterColor(2, random.random())

v_inputs_cpu = [Variable(i, requires_grad=True) for i in inputs]
fgf(*v_inputs_cpu).backward(grad_out)

v_inputs_gpu = [Variable(i.cuda(), requires_grad=True) for i in inputs]
fgf.cuda()(*v_inputs_gpu).backward(grad_out.cuda())

for v_cpu, v_gpu in zip(v_inputs_cpu, v_inputs_gpu):
    grad_cpu, grad_gpu = v_cpu.grad.data.cpu().numpy(), v_gpu.grad.data.cpu().numpy()
    assert np.isclose(grad_cpu, grad_gpu, rtol=1e-2).all()
print('\tGrad Check Result[GPU]: True')