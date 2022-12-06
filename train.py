import pytorch_lightning as pl
from model import MyModel
from dataset import MyDataModule

trainer = pl.Trainer(
    gpus=1,
    logger=pl.loggers.TensorBoardLogger(save_dir='tb_logs', name='MyModel'),
    max_epochs=10,
    log_every_n_steps=10,
    val_check_interval=10,
    # limit_val_batches=0.5

    callbacks=[pl.callbacks.ModelCheckpoint(
        filename='{val/loss:.2f}-{epoch}-{step}',
        monitor='val/loss',
        save_top_k=1,
        mode='min',
    )]
)

model = MyModel(lr_rate=0.0001)
datamodule = MyDataModule(transform=model.transform, dataset_dir='./dataset/Dataset-12', batch_size=16)

trainer.fit(
    model=model,
    datamodule=datamodule,
)

trainer.test(
    model=model,
    datamodule=datamodule,
)
