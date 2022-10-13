import pytorch_lightning as pl
from model import Extractor, MyModel
from dataset import MyDataModule

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MyModel'),
    max_epochs=1000,
    # log_every_n_steps=50,
    # val_check_interval=6,
    # limit_val_batches=0.5

    callbacks=[pl.callbacks.ModelCheckpoint(
        filename='{val/loss:.2f}-{epoch}-{step}',
        monitor='val/loss',
        save_top_k=1,
        mode='min',
    )]
)

extractor = Extractor()
model = MyModel(lr_rate=0.000001)
datamodule = MyDataModule(extractor=extractor, batch_size=128)

trainer.fit(
    model=model,
    datamodule=datamodule,
)

trainer.test(
    model=model,
    datamodule=datamodule,
)
