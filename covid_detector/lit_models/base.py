
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl


LR = 1e-4
OPTIMIZER = 'Adam'


class BaseLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: argparse.Namespace = None):
        super().__init__()
        self.model = model

        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get('lr', LR)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return {'optimizer': optimizer}

    def training_step(self, batch, batch_idx):
        x, target = batch
        out = self.model(x, target)
        self.log('loss', out['loss'])
        self.log('class_loss', out['class_loss'])
        self.log('box_loss', out['box_loss'])
        return out['loss']

    def validation_step(self, batch, batch_idx):
        x, target = batch
        out = self.model(x, target)
        self.log('loss', out['loss'])
        self.log('class_loss', out['class_loss'])
        self.log('box_loss', out['box_loss'])
        return out
