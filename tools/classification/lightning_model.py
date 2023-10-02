import lightning.pytorch as pl
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, net, num_of_category):
        super().__init__()
        self.net = net(num_of_category=num_of_category)
        self.train_accuracy = MulticlassAccuracy(num_classes=num_of_category)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_of_category)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_of_category)

    def training_step(self, batch, batch_idx):
        imgs = batch["image"]
        labels = batch["label"]

        predict = self.net(imgs)

        loss = F.cross_entropy(predict, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_accuracy(predict, labels)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["image"]
        labels = batch["label"]

        predict = self.net(imgs)

        loss = F.cross_entropy(predict, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        self.val_accuracy(predict, labels)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        imgs = batch["image"]
        labels = batch["label"]

        predict = self.net(imgs)

        loss = F.cross_entropy(predict, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self.test_accuracy(predict, labels)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        return optimizer
