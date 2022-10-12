import pytorch_lightning as pl
from model import MyModel
from dataset import MyDataModule
import torch
import numpy as np
from dataset import save_bvh_to_file

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MyModel')
)

# model = MyModel.load_from_checkpoint('pre_training/loss=119.99-epoch=81-step=491.ckpt')
model = MyModel()
data_module = MyDataModule(batch_size=128)

# 测试细节实现在 test_step 函数中
trainer.test(model=model, datamodule=data_module)
