import torch
from torch.nn import functional as F
from torch import nn
import torchmetrics
import pytorch_lightning as pl
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class MyModel(pl.LightningModule):
    def __init__(self, lr_rate=0.01):
        super().__init__()

        # 学习率
        self.lr_rate = lr_rate

        # 预训练的特征提取器
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=12)

        # # 输出模型
        # print(self.backbone)

        # # 冻结backbone模型所有参数
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # # 解冻最后一个head层参数
        # for param in self.backbone.head.parameters():
        #     param.requires_grad = True

        # # 替换最后一个head层
        # self.backbone.head = nn.Sequential(
        #     nn.Linear(768, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 12)
        # )
        
        # 图片预处理变换器
        self.transform = create_transform(**resolve_data_config({}, model=self.backbone))

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 度量标准
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.backbone(x) # [batch_size, 3, 224, 224] -> [batch size, num_classes]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 图片 x: [batch_size, 3, 224, 224]
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
