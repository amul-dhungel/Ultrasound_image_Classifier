"""
    Author: Mauro Mendez.
    Date: 22/11/2021.

    This file creates the model object which contains the minimal settings to train
    and test a model using pytorch lightning
"""

import pytorch_lightning as pl
import torchmetrics
import torch

from hyperparameters import parameters as params
import models


class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # Model
        if params['model'] == "resnet18":
            self.model = models.resnet18()
        elif params['model'] == "efficientnet":
            self.model = models.efficientnet()

        # Loss Function
        if params['criterion'] == "BCEWithLogitsLoss":
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.transf = torch.nn.Sigmoid()
        elif params['criterion'] == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss()
            self.transf = torch.nn.Softmax()

        # Metrics
        self.epoch_ = 100
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_sens = torchmetrics.Recall()
        self.test_spec = torchmetrics.Specificity()
        self.test_auc = torchmetrics.AUROC(pos_label=1)

    def forward(self, x):
        x = self.model(x)
        return x

    # convenient method to get the loss from a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = self.criterion(logits, ys.reshape(-1, 1))
        return logits, loss

    # What to do for a training batch
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = self.transf(logits).reshape(-1)
        ys = torch.as_tensor(ys, dtype=torch.int)

        self.log('train/loss', loss, on_epoch=True)
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    # What to do for a training batch
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = self.transf(logits).reshape(-1)
        ys = torch.as_tensor(ys, dtype=torch.int)

        self.log('val/loss', loss, on_epoch=True)
        self.val_acc(preds, ys)
        self.log('val/acc', self.val_acc, on_epoch=True)

    # Pytorch ligthning optimizer configuration
    def configure_optimizers(self):
        if params['optimizer'] == 'ADAM':
            optimizer = torch.optim.Adam(self.parameters(), lr=params['lear_rate'])
        elif params['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=params['lear_rate'], momentum=0.9)

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=0.001),
            'monitor': 'val/loss'
        }

    # What to do for a testing batch
    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = self.transf(logits).reshape(-1)
        ys = torch.as_tensor(ys, dtype=torch.int)

        self.test_acc(preds, ys)
        self.test_sens(preds, ys)
        self.test_spec(preds, ys)
        self.test_auc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/sens_epoch", self.test_sens, on_step=False, on_epoch=True)
        self.log("test/spec_epoch", self.test_spec, on_step=False, on_epoch=True)
        self.log("test/auc_epoch", self.test_auc, on_step=False, on_epoch=True)

    # Method to freeze the weights of the model
    # def freeze_weights(self):
    #     for param in list(self.model.parameters())[:-50]:
    #         param.requires_grad = False
