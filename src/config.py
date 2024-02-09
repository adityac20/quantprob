"""
Basic Configurations
"""

import os
import pathlib

# PATH Variables
DATA_DIR = "~/Documents/Datasets_Global/"
PRETRAINED_DIR = "./models/pretrained/"
CHECKPOINT_DIR = "./checkpoints/"
DUMP_DIR = "./data/interim/"
RESULTS_DIR = "./reports/"
LOGS_DIR = "./tb_logs/"

# TRAINING PARAMS
BATCHSIZE = 1024
EPOCHS = 1000
CHECK_VAL_EVERY_N_EPOCHS = 1
NUM_WORKERS = 14

# COMPUTE PARAMS
ACCERLATOR = "gpu"
DEVICES = 2
PRECISION = 32

# Quantile Parameters
NUM_QUANTILES = 100

# POSSIBLE NAMES FOR PRETRAINED MODELS
PRETRAINED_MODELS = [
    "resnet20_cifar10",
    "resnet20_cifar100",
    "densenet_cifar10",
    "densenet_cifar100",
    "densenet_svhn",
    "resnet34_cifar10",
    "resnet34_cifar100",
    "resnet34_svhn",
]
