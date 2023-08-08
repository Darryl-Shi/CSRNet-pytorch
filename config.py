import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''
    def __init__(self):
        self.dataset_root = 'data/part_A_final', 'data/part_B_final', 'data/mall_dataset'
        self.batch_size   = 1                   # batch size
        self.epochs       = 2000                # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.__mkdir(self.checkpoints)
        self.writer       = SummaryWriter() 
        self.project      = 'crowd-counting'

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)