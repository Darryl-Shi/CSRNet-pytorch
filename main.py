import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
import wandb
import os

from config import Config
from model import CSRNet
from dataset import create_train_dataloader, create_test_dataloader
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from utils import denormalize
import matplotlib.pyplot as plt
import matplotlib.cm as CM

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

class CSRNetLightning(pl.LightningModule):
    def __init__(self, config, lr):
        super().__init__()
        self.config = config
        self.lr = lr
        self.model = CSRNet()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, gt_densitymap = batch['image'], batch['densitymap']
        et_densitymap = self(image)
        loss = self.criterion(et_densitymap, gt_densitymap)
        self.log('train_loss', loss)
        cfg.writer.add_scalar('train_loss', loss, str(self.current_epoch))
        return loss

    def validation_step(self, batch, batch_idx):
        image, gt_densitymap = batch['image'], batch['densitymap']
        et_densitymap = self(image).detach()
        mae = abs(et_densitymap.data.sum() - gt_densitymap.data.sum())
        self.log('val_mae', mae)
        cfg.writer.add_scalar('val_loss', mae, self.current_epoch)
        if batch_idx == 0:
            cfg.writer.add_image(str(self.current_epoch)+'/Image', denormalize(image[0].cpu()))
            cfg.writer.add_image(str(self.current_epoch)+'/Estimate density count:'+ str('%.2f'%(et_densitymap[0].cpu().sum())), et_densitymap[0]/torch.max(et_densitymap[0]))
            cfg.writer.add_image(str(self.current_epoch)+'/Ground Truth count:'+ str('%.2f'%(gt_densitymap[0].cpu().sum())), gt_densitymap[0]/torch.max(gt_densitymap[0]))
        return mae
    
    def predict_step(self, batch, batch_idx):
        image = batch['image']
        et_densitymap = self(image).detach()
        et_densitymap = et_densitymap.squeeze(0).squeeze(0).cpu().numpy()
        return et_densitymap

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return create_train_dataloader(self.config.dataset_root, use_flip=True, batch_size=self.config.batch_size)

    def val_dataloader(self):
        return create_test_dataloader(self.config.dataset_root)


import os
if __name__ == "__main__":
    cfg = Config()
    
    if cfg.train:
        wandb.init()    
        logger=WandbLogger(project=cfg.project)
        def objective(trial: optuna.trial.Trial) -> float:
            # We optimize the number of layers, hidden units in each layer and dropouts.
            dropout = trial.suggest_float("dropout", 0.2, 0.5)
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

            model = CSRNetLightning(cfg, dropout, lr)

            trainer = pl.Trainer(
                logger=True,
                limit_val_batches=10,
                enable_checkpointing=False,
                max_epochs=3,
                accelerator="auto",
                devices="auto",
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_mae")],
                gradient_clip_val=0.5,
                gradient_clip_algorithm="value",
            )
            hyperparameters = dict(dropout=dropout, lr=lr)
            trainer.logger.log_hyperparams(hyperparameters)
            trainer.fit(model)

            return trainer.callback_metrics["val_mae"].item()
        
        if os.environ.get("ENV") != "TEST":
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=cfg.checkpoints,
                filename='{epoch}-{val_mae:.2f}',
                monitor='val_mae',
                mode='min',
                save_top_k=1,
            )
        if cfg.sweep:
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(direction="minimize", pruner=pruner)
            study.optimize(objective, n_trials=100)

            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
        else:
            model = CSRNetLightning(cfg, 1e-4)
            trainer = pl.Trainer(
                logger=logger,
                limit_val_batches=10,
                enable_checkpointing=True,
                max_epochs=cfg.epochs,
                accelerator="auto",
                devices="auto",
                callbacks=[checkpoint_callback],
                gradient_clip_val=0.5,
                gradient_clip_algorithm="value",
            )
            trainer.fit(model)
            trainer.test()

    else:
        # Deprecated
        # data_loader = create_test_dataloader('data/part_B_final')
        # model = CSRNetLightning.load_from_checkpoint('checkpoints/epoch=156-val_mae=30.95.ckpt', config=cfg, lr=1e-4, map_location='cpu')
        # trainer = pl.Trainer()
        # preds = trainer.predict(model, data_loader)
        # predictions, gt = preds[0][0], preds[0][1]
        # torch.save(predictions, 'output/predictions.pt')
        # plt.imsave('output/test.png', predictions, cmap=CM.jet)
        # plt.imsave('output/gtdm.png', gt, cmap=CM.jet)


    
        pass