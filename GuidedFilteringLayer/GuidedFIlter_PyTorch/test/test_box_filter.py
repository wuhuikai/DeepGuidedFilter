import time

import torch
import numpy as np

from skimage import img_as_float
from skimage.io import imread

from torch.autograd import gradcheck, Variable

from guided_filter_pytorch.box_filter import BoxFilter

## BoxFilter
print('BoxFilter:')
# check forward
y = BoxFilter(3)(Variable(torch.from_numpy(np.reshape(np.arange(1, 73), (1, 1, 8, 9))).cuda())).data.cpu().numpy()
assert np.isclose(y.mean(),  1137.6,  0.1)
assert np.isclose(y.std(),    475.2,  0.1)
print('\tForward passed')
# forward on img
im = Variable(torch.from_numpy(np.transpose(img_as_float(imread('test/rgb.jpg')), (2, 0, 1))[None]).float().cuda())
start_time = time.time()
r = BoxFilter(64)(im)
end_time   = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time-start_time))

r = r.data.cpu().numpy().squeeze().transpose(1, 2, 0)
assert np.isclose(r[:,:,0].mean(), 10305.0, 0.1)
assert np.isclose(r[:,:,0].std(),   2206.4, 0.1)
assert np.isclose(r[:,:,1].mean(),  7536.0, 0.1)
assert np.isclose(r[:,:,1].std(),   2117.0, 0.1)
assert np.isclose(r[:,:,2].mean(),  6203.0, 0.1)
assert np.isclose(r[:,:,2].std(),   2772.3, 0.1)
print('\tPassed ...')

# backward | check grad
test = gradcheck(BoxFilter(3), (Variable(torch.rand((3, 5, 8, 9)).double().cuda(), requires_grad=True), ), eps=1e-6, atol=1e-4)
print('\tGrad Check Result:', test)