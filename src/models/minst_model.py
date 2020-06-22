import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):

    def __init__(self):
        super(MNISTModel, self).__init__()
        self.linear1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        # called with self(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        return x

    def training_step(self, batch, step):
        # implements a single training pass
        feat, label = batch
        pred = self(feat)
        loss = F.cross_entropy(pred, label)
        tb_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, step):
        # implements a single validation pass
        feat, label = batch
        pred = self(feat)
        loss = F.cross_entropy(pred, label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # calculating loss across steps in an epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tb_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tb_logs}

    def test_step(self, batch, step):
        # implements a single training pass
        feat, label = batch
        pred = self(feat)
        loss = F.cross_entropy(pred, label)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # calculating loss across steps in an epoch
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tb_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tb_logs, 'progress_bar': tb_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32) 



    