import os

from tqdm import tqdm

from tensorboardX import SummaryWriter

class VisUtils(object):
    def __init__(self, name, n_iter, n_epoch, log_dir='tensorboard_logs', stat_dir='tensorboard_stats'):
        self.n_iter = n_iter
        self.stat_dir = os.path.join(stat_dir, name)
        if not os.path.isdir(self.stat_dir):
            os.makedirs(self.stat_dir)

        self.e_bar = tqdm(total=n_epoch, desc='#Epoch')
        self.i_bar = tqdm(total=n_iter,  desc='  #Iter')

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, name))

    def reset(self, n_iter, n_epoch):
        self.e_bar.close()
        self.i_bar.close()
        self.e_bar = tqdm(total=n_epoch, desc='#Epoch')
        self.i_bar = tqdm(total=n_iter,  desc='  #Iter')


    def update(self, post_fix):
        self.i_bar.set_postfix(**post_fix)
        self.i_bar.update()

    def next_epoch(self):
        self.e_bar.update()

        self.i_bar.close()
        self.i_bar = tqdm(total=self.n_iter, desc='  #Iter')

    def close(self):
        self.writer.export_scalars_to_json(os.path.join(self.stat_dir, 'scalars.json'))
        self.writer.close()