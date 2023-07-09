import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl

from config import Config
from model import CSRNet
from dataset import create_train_dataloader, create_test_dataloader
from utils import denormalize


class CSRNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CSRNet()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, gt_densitymap = batch['image'], batch['densitymap']
        et_densitymap = self(image)
        loss = self.criterion(et_densitymap, gt_densitymap)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, gt_densitymap = batch['image'], batch['densitymap']
        et_densitymap = self(image).detach()
        mae = abs(et_densitymap.data.sum() - gt_densitymap.data.sum())
        self.log('val_mae', mae)
        return mae

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def train_dataloader(self):
        return create_train_dataloader(self.config.dataset_root, use_flip=True, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return create_test_dataloader(self.config.dataset_root)


if __name__ == "__main__":
    cfg = Config()
    model = CSRNetLightning(cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.checkpoints,
        filename='{epoch}-{val_mae:.2f}',
        monitor='val_mae',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(max_epochs=cfg.epochs, accelerator="gpu", callbacks=[checkpoint_callback])
    trainer.fit(model)