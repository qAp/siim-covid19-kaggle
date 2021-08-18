
import argparse
import torch.nn as nn
import pytorch_lightning as pl


class BaseLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        class_out, box_out = self.model(x)
        loss = class_out[0].sum()
        return loss