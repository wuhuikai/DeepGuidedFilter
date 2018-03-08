import os
import time

import torch

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import Config
from dataset import SuDataset
from vis_utils import VisUtils


default_config = Config(
    TASK = None,
    NAME = 'LR',
    N_START = 0,
    N_EPOCH = 150,
    DATA_SET = 512,
    FINE_SIZE = -1,
    #################### CONSTANT #####################
    IMG = 'dataset',
    SAVE = 'checkpoints',
    LIST = 'train_test_list',
    BATCH = 1,
    SHOW_INTERVEL = 64,
    N_PROCESS = 4,
    LOW_SIZE = 64,
    GPU = 0,
    LR = 0.0001,
    # clip
    clip = None,
    # model
    model = None,
    # forward
    forward = None,
    # img size
    exceed_limit = None,
    # vis
    vis = None
)

def run(config, keep_vis=False):
    assert config.TASK is not None, 'Please set task name: TASK'

    save_path = os.path.join(config.SAVE, config.TASK, config.NAME)
    path = os.path.join(save_path, 'snapshots')
    if not os.path.isdir(path):
        os.makedirs(path)

    # data set
    train_data = SuDataset(config.IMG,
                           os.path.join(config.LIST,
                                        config.TASK,
                                        'train_{}.csv'.format(config.DATA_SET)),
                           low_size=config.LOW_SIZE,
                           fine_size=config.FINE_SIZE)
    train_loader = DataLoader(train_data, batch_size=config.BATCH, shuffle=True, num_workers=config.N_PROCESS)

    # loss
    criterion = nn.MSELoss()

    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            config.model.cuda()
            criterion.cuda()

    # setup optimizer
    optimizer = optim.Adam(config.model.parameters(), lr=config.LR)

    if config.vis is None:
        config.vis = VisUtils(os.path.join(config.TASK, config.NAME), len(train_loader), config.N_EPOCH)
    else:
        config.vis.reset(len(train_loader), config.N_EPOCH)
    for epoch in range(config.N_START, config.N_START+config.N_EPOCH):
        total_loss = 0
        for idx, imgs in enumerate(train_loader):
            if config.exceed_limit is not None and config.exceed_limit(imgs[0].size()[2:]):
                config.vis.update({})
                continue

            t = time.time()

            y, gt = config.forward(imgs, config)
            loss = criterion(y, Variable(gt))
            # backward
            optimizer.zero_grad()
            loss.backward()
            if config.clip is not None:
                torch.nn.utils.clip_grad_norm(config.model.parameters(), config.clip)
            optimizer.step()

            ##################### PLOT/SHOW ######################
            global_step = epoch*len(train_loader)+idx

            loss = loss.data.cpu()[0] * 255 * 255
            total_loss += loss

            config.vis.writer.add_scalars('loss', {
                'instance_loss': loss,
                'mean_loss': total_loss/(idx+1)
            }, global_step=global_step)
            config.vis.writer.add_scalar('time', time.time()-t, global_step=global_step)
            config.vis.update({'loss': total_loss/(idx+1), 'time': time.time()-t})

            if idx % (config.SHOW_INTERVEL//config.BATCH) == 0:
                for name, param in config.model.named_parameters():
                    if param.grad is not None:
                        config.vis.writer.add_scalar('grad/'+name, param.grad.data.norm(2), global_step=global_step)

        # save
        torch.save(config.model.state_dict(), os.path.join(save_path, 'snapshots', 'net_epoch_{}.pth'.format(epoch)))
        torch.save(config.model.state_dict(), os.path.join(save_path, 'snapshots', 'net_latest.pth'))
        config.vis.next_epoch()

    if not keep_vis:
        config.vis.close()