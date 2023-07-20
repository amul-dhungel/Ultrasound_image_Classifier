"""
    Author: Mauro Mendez.
    Date: 22/11/2021.

    This file run the training and testing of the model.
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from hyperparameters import parameters as params
from network import Net
from dataset import CustomDataModule



pl.seed_everything(params['seed'])

name = 'Baseline'

wandb_logger = WandbLogger(project="efficinetNet-b7", name=name)

# setup data
dataset = CustomDataModule(aug=True)
dataset.setup() 

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val/acc',
    dirpath='../weights/',
    filename=name+'_weights-{epoch:02d}',
    mode='max',
)
early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.000001, patience=10, verbose=False, mode="min")

trainer = pl.Trainer(
    logger=wandb_logger,    # W&B integration
    log_every_n_steps=2,   # set the logging frequency
    min_epochs=params['min_epochs'],
    max_epochs=params['epochs'],
    precision=params['precision'],
    accumulate_grad_batches=params['accumulate_grad_batches'],
    gpus=-1,                # use all GPUs
    deterministic=True,      # keep it deterministic
    callbacks=[checkpoint_callback]#, early_stop_callback]
)



if __name__ == '__main__':    
# setup model
    model = Net()
#model.freeze_weights()

# fit the model
    trainer.fit(model, dataset)

# evaluate the model on a test set

    trainer.test(model,datamodule=dataset, ckpt_path=checkpoint_callback.best_model_path,verbose = True) # uses best model


