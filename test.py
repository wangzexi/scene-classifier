import pytorch_lightning as pl
from model import MyModel
from dataset import MyDataModule

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MyModel')
)

model = MyModel.load_from_checkpoint(r'tb_logs\MyModel\version_4\checkpoints\val\loss=0.32-epoch=9-step=500.ckpt')
# model = MyModel()
datamodule = MyDataModule(transform=model.transform, dataset_dir='./dataset/Dataset-12', batch_size=16)


# 测试细节实现在 test_step 函数中
trainer.test(model=model, datamodule=datamodule)
