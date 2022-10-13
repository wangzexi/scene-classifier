import pytorch_lightning as pl
from model import Extractor, MyModel
from dataset import MyDataModule

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MyModel')
)

extractor = Extractor()
model = MyModel.load_from_checkpoint(r'tb_logs\MyModel\version_3\checkpoints\val\loss=0.13-epoch=585-step=46880.ckpt')
# model = MyModel()
datamodule = MyDataModule(extractor, batch_size=128)

# 测试细节实现在 test_step 函数中
trainer.test(model=model, datamodule=datamodule)
