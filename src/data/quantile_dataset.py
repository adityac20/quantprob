import pdb
import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

import torchvision
from torchvision import transforms, datasets

import pytorch_lightning as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import src.config as config
from src.utils.utils_train import get_pretrained_model, get_base_datasets


def _weighted_quantile(arr, quantiles, weights):
    """ """
    indsort = np.argsort(arr)
    weights_cum = np.cumsum(weights[indsort])
    weights_cum = weights_cum / np.sum(weights)
    return np.interp(quantiles, weights_cum, arr[indsort])


def _get_class_weights(ylabel, class_no):
    """ """
    weights = np.ones(len(ylabel))
    weights[ylabel == class_no] = np.sum(ylabel != class_no)
    weights[ylabel != class_no] = np.sum(ylabel == class_no)
    return weights


def weighted_quantiles(logits, quantiles, labels):
    """
    - This is equivalent to 'lower' interpolation in torch.quantile.
    """

    quantiles_data = []
    for class_no in range(logits.shape[1]):
        weights = _get_class_weights(labels, class_no)
        quantiles_data.append(
            _weighted_quantile(logits[:, class_no], quantiles, weights)
        )
    quantiles_data = np.stack(quantiles_data, axis=1)

    return quantiles_data


def _compute_and_cache_quantile_labels(model, train_dataset, filename):
    """
    - Compute the quantile labels for the pretrained model.
    - Note that we use weighted quantiles for multi-class classification.
    """

    """
    STEP 1: Extract the features from the pretrained model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.fc = torch.nn.Identity()  # Remove the last layer
    model.to(device)
    model.eval()
    features, labels = [], []
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCHSIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    """
    STEP 2: Train a One-Vs-Rest classifier for each class.
    """
    clf = LogisticRegression(max_iter=1000, multi_class="ovr")
    clf.fit(features, labels)
    print("(check) accuracy of logstic regression:", clf.score(features, labels))
    logits = clf.decision_function(features)

    """
    STEP 3: Compute the quantile labels by computing weigted quantiles on the logits.
    """
    quantiles_list = np.linspace(0, 1, config.NUM_QUANTILES + 2)[1:-1]
    quantiles = weighted_quantiles(logits, quantiles_list, labels)
    quant_labels = (logits[:, np.newaxis, :] > quantiles[np.newaxis, :, :]) * 1

    # sanity check for the quantile labels
    pred_quant_labels = np.argmax(np.mean(quant_labels, axis=1), axis=1)
    print("(check) accuracy of quantile labels :", np.mean(pred_quant_labels == labels))

    """
    STEP 4: Cache the quantile labels for future use.
    """
    torch.save(quant_labels, filename)
    print(f"Quantile labels for are cached at {filename}.")


def cache_quantile_labels(
    base_model, base_dataset, quant_model_name, force_compute_cache
):
    """ """
    filename = os.path.join(config.DUMP_DIR, f"{quant_model_name}_quantile_labels.npy")

    if force_compute_cache:
        _compute_and_cache_quantile_labels(base_model, base_dataset, filename)

    if os.path.exists(filename):
        print(f"Quantile labels for {quant_model_name} already exist.")
    else:
        _compute_and_cache_quantile_labels(base_model, base_dataset, filename)


class QuantileDataset(Dataset):
    """
    - Appends the input with a random quantile value.
    - Along with the target, it also returns the quantile label.
    - Note that quantile labels are multi-label vectors.
    """

    def __init__(
        self,
        base_model,
        base_dataset,
        quant_model_name,
        train=True,
        random_quantile=False,
        force_compute_cache=False,
    ):
        super().__init__()
        self.dataset = base_dataset
        self.train = train
        self.random_quantile = random_quantile
        cache_quantile_labels(
            base_model,
            base_dataset,
            quant_model_name,
            force_compute_cache=force_compute_cache,
        )
        self.num_classes = 10

        self.quantile_labels = torch.from_numpy(
            torch.load(
                os.path.join(config.DUMP_DIR, f"{quant_model_name}_quantile_labels.npy")
            )
        ).float()
        self.quantiles_list = torch.linspace(0, 1, config.NUM_QUANTILES + 2)[1:-1]
        rng = np.random.default_rng(42)
        self.ind_random = rng.permutation(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label = self.dataset.__getitem__(idx)

        assert x.dim() == 3, "Only Implemented for RGB-Images"
        x = x.unsqueeze(0)
        x = x.repeat(config.NUM_QUANTILES, 1, 1, 1)
        # shape of x: (num_quantiles, 3, 32, 32) for CIFAR-10
        quant_val = self.quantiles_list.view(-1, 1, 1, 1) * torch.ones((1, 1, 32, 32))
        x = torch.cat([x, quant_val], dim=1)
        label = label * torch.ones(config.NUM_QUANTILES).long()

        if self.train:
            y = (self.quantile_labels[idx, :, :]).float()
            return x, y, label
        else:
            out = F.one_hot(torch.tensor([label]).long(), self.num_classes).float()
            return x, out, label
