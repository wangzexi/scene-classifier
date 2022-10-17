import torch
from torch.nn import functional as F
from torch import nn
import torchmetrics
import pytorch_lightning as pl
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        # 预训练的图片特征提取器
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = torch.nn.Identity()

        self.transform = create_transform(**resolve_data_config({}, model=self.model))

    def forward(self, x):
        return self.model(x)  # [batch_size, 768]


class MyModel(pl.LightningModule):
    def __init__(self, lr_rate=0.01):
        super().__init__()

        self.lr_rate = lr_rate
        self.fc = nn.Linear(768, 6)  # [batch_size, 768] -> [batch_size, 6]
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.fc(x)  # [batch_size, 768] -> [batch_size, 6]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 图片特征 x: [batch_size, 768]
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
