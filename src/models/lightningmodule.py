import pdb
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

import lightning.pytorch as pl
import torchmetrics
from torchmetrics import Metric

import src.config as config
from src.models.densenet import DenseNet3
from src.models.resnet import ResNet34
from src.utils.utils_train import get_pretrained_model, get_base_datasets

import calibration as cal


class QuantileNetwork(pl.LightningModule):
    """ """

    def __init__(self, name_base_model: str, **kwargs: Any):
        super().__init__()
        self.backbone, num_classes, size_dataset = get_pretrained_model(
            name_base_model, use_pretrained=True
        )

        self.save_hyperparameters()

        # Change the first conv layer to accept 1 additional input
        self.backbone.conv1 = nn.Conv2d(
            in_channels=self.backbone.conv1.in_channels + 1,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # Metrics
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes)
        self.accuracy_multiclass = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        # Saving logits and y for each epoch to compute the metrics
        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.valid_step_outputs_quant = []
        self.test_step_outputs_quant = []

        # Quantile params
        self.num_classes = num_classes
        self.size_dataset = size_dataset
        self.num_quant_rep = config.NUM_QUANTILES
        self.quantiles_list = nn.Parameter(
            torch.linspace(0, 1, self.num_quant_rep + 2)[1:-1]
        )

    def on_train_start(self) -> None:
        # log hyperparams
        self.logger.log_hyperparams(
            self.hparams,
            {
                "train_acc": 0,
                "val_acc": 0,
                "val_acc_quant": 0,
                "val_calib_error": 0,
                "val_loss": 0,
                "test_loss": 0,
                "test_acc": 0,
                "test_acc_quant": 0,
                "test_calib_error": 0,
            },
        )

    def forward(self, x):
        """ """
        return self.backbone(x)

    def _common_step(self, batch, batch_idx):
        """ """
        x, yquant, _ = batch  # ytrue is not used for training
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, yquant)
        return loss, logits, yquant

    def training_step(self, batch, batch_idx):
        loss, logits, yquant = self._common_step(batch, batch_idx)
        self.train_step_outputs.append(
            {"logits": logits, "yquant": yquant, "loss": loss}
        )
        return loss

    def on_train_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.train_step_outputs], dim=0)
        yquant = torch.cat([x["yquant"] for x in self.train_step_outputs], dim=0)
        train_loss_epoch = torch.stack(
            [x["loss"] for x in self.train_step_outputs]
        ).mean()
        self.log_dict(
            {
                "train_acc": self.accuracy(logits, yquant),
                "train_loss_epoch": train_loss_epoch,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, _, ytrue = batch
        loss, logits, yquant = self._common_step(batch, batch_idx)
        self.valid_step_outputs.append(
            {"logits": logits, "yquant": yquant, "loss": loss, "ytrue": ytrue}
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        # Standard Metrics - Accuracy and BCE-Loss
        logits = torch.cat([x["logits"] for x in self.valid_step_outputs], dim=0)
        yquant = torch.cat([x["yquant"] for x in self.valid_step_outputs], dim=0)
        ytrue = torch.cat([x["ytrue"] for x in self.valid_step_outputs], dim=0)
        val_loss_epoch = torch.stack(
            [x["loss"] for x in self.valid_step_outputs]
        ).mean()

        preds = ((logits > 0) * 1).float()

        preds = preds.reshape(-1, self.num_quant_rep, self.num_classes)
        prob_quantile = torch.mean(preds, dim=1)

        pred_quantile = torch.argmax(prob_quantile, dim=1)

        ytrue = ytrue.reshape(-1, self.num_quant_rep)
        ytrue_val, _ = torch.max(ytrue, dim=1)

        prob_quantile_cpu = prob_quantile.clone().cpu().detach().numpy()
        ytrue_cpu = ytrue_val.clone().cpu().detach().numpy()
        val_calbration_error = cal.get_calibration_error(
            prob_quantile_cpu, ytrue_cpu, mode="top-label"
        )

        self.log_dict(
            {
                "val_acc": self.accuracy(logits, yquant),
                "val_loss_epoch": val_loss_epoch,
                "val_acc_quant": self.accuracy_multiclass(prob_quantile, ytrue_val),
                "val_calib_error": val_calbration_error,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.valid_step_outputs.clear()
        self.valid_step_outputs_quant.clear()

    def test_step(self, batch, batch_idx):
        """
        - _common_step is meaningless for the test step since we do not have pre-trained outputs
            for the test.
        """
        x, _, ytrue = batch
        loss, logits, yquant = self._common_step(batch, batch_idx)
        self.test_step_outputs.append(
            {"logits": logits, "yquant": yquant, "loss": loss, "ytrue": ytrue}
        )
        return loss

    def on_test_epoch_end(self) -> None:

        # Standard Metrics - Accuracy and BCE-Loss
        logits = torch.cat([x["logits"] for x in self.test_step_outputs], dim=0)
        yquant = torch.cat([x["yquant"] for x in self.test_step_outputs], dim=0)
        ytrue = torch.cat([x["ytrue"] for x in self.test_step_outputs], dim=0)
        test_loss_epoch = torch.stack(
            [x["loss"] for x in self.test_step_outputs]
        ).mean()

        preds = ((logits > 0) * 1).float()
        preds = preds.reshape(-1, self.num_quant_rep, self.num_classes)
        prob_quantile = torch.mean(preds, dim=1)
        pred_quantile = torch.argmax(prob_quantile, dim=1)

        ytrue = ytrue.reshape(-1, self.num_quant_rep)
        assert torch.allclose(
            torch.zeros_like(ytrue), torch.std(ytrue, dim=1, keepdim=True)
        ), "ytrue is not constant across the quantiles"
        ytrue = torch.argmax(ytrue, dim=1)

        test_calbration_error = cal.get_calibration_error(
            prob_quantile, ytrue, mode="top-label"
        )

        self.log_dict(
            {
                "test_acc": self.accuracy(logits, yquant),
                "test_loss_epoch": test_loss_epoch,
                "test_acc_quant": self.accuracy_multiclass(prob_quantile, ytrue),
                "test_calib_error": test_calbration_error,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.test_step_outputs.clear()
        self.test_step_outputs_quant.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
