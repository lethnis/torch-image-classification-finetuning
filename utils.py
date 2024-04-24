import torch
from torch import optim, nn
from torch.nn import functional as F

from torchvision import models

from torchmetrics import Accuracy
import lightning as L


class LitWrapper(L.LightningModule):
    def __init__(self, model):

        super().__init__()

        self.model = model
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, x, y):
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        return loss, pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, pred = self._shared_step(x, y)
        self.train_accuracy.update(pred, y)
        self.log_dict({"train_loss": loss, "train_acc": self.train_accuracy}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, pred = self._shared_step(x, y)
        self.val_accuracy.update(pred, y)
        self.log_dict({"val_loss": loss, "val_acc": self.val_accuracy}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, pred = self._shared_step(x, y)
        self.test_accuracy.update(pred, y)
        self.log_dict({"test_loss": loss, "test_acc": self.test_accuracy}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


def get_lit_efficientnet():
    # load pre-trained model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # freeze feature extractor
    model.features.requires_grad_(False)
    # change base classifier
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 10))
    # create model as lightning module
    return LitWrapper(model)


def get_lit_shufflenet_0_5():
    # load pre-trained model
    model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    # freeze the whole model
    model.requires_grad_(False)
    # make new classifier, it will be trainable by default
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 10))
    # create model as lightning module
    return LitWrapper(model)


def get_lit_shufflenet_1_0():
    # load pre-trained model
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    # freeze the whole model
    model.requires_grad_(False)
    # make new classifier, it will be trainable by default
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, 10))
    # create model as lightning module
    return LitWrapper(model)
