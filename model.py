import torch
from torch.nn import functional as F
from torch import nn
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt


class MyModel(pl.core.lightning.LightningModule):
    def __init__(self, lr_rate=0.01):
        super().__init__()

        self.lr_rate = lr_rate
        self.fc = nn.Linear(1000, 6)  # [1, 1000] -> [1, 6]
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.fc(x)  # [batch_size, 1000] -> [batch_size, 6]
        # x = torch.softmax(x, dim=1) # [batch_size, 6] -> [batch_size, 6]
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 图片特征 x: [batch_size, 1000]
        # 分类索引 y: [batch_size]

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        self.train_acc(F.softmax(logits, dim=-1), y)
        self.log_dict({'train/loss': loss, 'train/acc': self.train_acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.val_acc(F.softmax(logits, dim=-1), y)
        self.log_dict({'val/loss': loss, 'val/acc': self.val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.test_acc(F.softmax(logits, dim=-1), y)
        self.log_dict({'test/loss': loss, 'test/acc': self.test_acc})
