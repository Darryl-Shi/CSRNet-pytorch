import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = 'data/part_A_final', 'data/part_B_final', 'data/mall_dataset'
        self.lr           = 1e-5                # learning rate
        self.batch_size   = 1                   # batch size
        self.epochs       = 2000                # epochs
        self.checkpoints  = '/mnt/disks/persist/checkpoints/pw'     # checkpoints dir
        self.project      = 'crowd-counting'
        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)