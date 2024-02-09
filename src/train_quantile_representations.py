import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import re
import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append("../models/")

import torch

import torchvision
from torchvision import transforms, datasets

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner


from src.models.lightningmodule import QuantileNetwork
from src.data.datamodule import DataModule

from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import src.config as config

# To stop lightning from complaining about precision
torch.set_float32_matmul_precision("high")
name_base_model = "resnet34_cifar10"
exp_number = "{:02d}".format(1)

logdir = os.path.abspath(f"./tb_logs/BigExp{exp_number}/")
if not os.path.exists(logdir):
    os.makedirs(logdir)
modeldir = os.path.abspath(f"./models/BigExp{exp_number}/")
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

assert (
    name_base_model in config.PRETRAINED_MODELS
), f"Model {name_base_model} not implemented."
current_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

quant_model = QuantileNetwork(name_base_model)
quant_datamodule = DataModule(name_base_model)

logger = TensorBoardLogger(
    logdir,
    name=f"{name_base_model}" + current_time,
    version=0,
    default_hp_metric=False,
)
loss_callback = ModelCheckpoint(
    dirpath=modeldir + f"/{name_base_model}" + current_time + "/",
    filename=f"{name_base_model}" + "{epoch:03d}-{val_calib_error:.4f}",
    save_top_k=1,
    monitor="val_calib_error",
    mode="min",
)
acc_callback = ModelCheckpoint(
    dirpath=modeldir + f"/{name_base_model}" + current_time + "/",
    filename=f"{name_base_model}" + "{epoch:03d}-{val_acc_quant:.4f}",
    save_top_k=1,
    monitor="val_acc_quant",
    mode="max",
)
early_stop_callback = EarlyStopping(
    monitor="val_acc_quant", patience=20, mode="max", verbose=True
)

trainer = pl.Trainer(
    strategy="ddp_find_unused_parameters_true",
    logger=logger,
    accelerator=config.ACCERLATOR,
    devices=[0, 1],
    min_epochs=1,
    max_epochs=100,
    precision=config.PRECISION,
    callbacks=[loss_callback, early_stop_callback, acc_callback],
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    max_time="00:24:00:00",
    fast_dev_run=False,
    num_sanity_val_steps=0,
    accumulate_grad_batches=200,
)

trainer.fit(quant_model, datamodule=quant_datamodule)
# Set the device to the first GPU
trainer.test(quant_model, datamodule=quant_datamodule)
