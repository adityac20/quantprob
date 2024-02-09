import pdb
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler

import lightning.pytorch as pl

import src.config as config
from src.data.quantile_dataset import QuantileDataset, cache_quantile_labels
from src.utils.utils_train import get_pretrained_model, get_base_datasets


def manual_collate_fn(data):
    """
    - This is a manual collate function to avoid the default collate function
      from torch.utils.data.DataLoader.
    - This is necessary because the default collate function from DataLoader
      does not work with the QuantileDataset.
    """
    features = torch.cat([item[0] for item in data], dim=0)
    targets = torch.cat([item[1] for item in data], dim=0)
    gt_labels = torch.cat([item[2] for item in data], dim=0)
    return features, targets, gt_labels


class DataModule(pl.LightningDataModule):
    """ """

    def __init__(self, name_base_model: str) -> None:
        super().__init__()

        self.name_base_model = name_base_model
        self.num_quantiles = config.NUM_QUANTILES
        self.quantile_list = np.linspace(0, 1, self.num_quantiles + 2)[1:-1]
        self.batch_size = 10

        base_model, num_classes, _ = get_pretrained_model(name_base_model)
        self.model = base_model
        self.num_classes = num_classes

    def prepare_data(self) -> None:
        total_ds, _, _, test_transform = get_base_datasets(self.name_base_model)
        train_ds, val_ds = random_split(
            total_ds,
            [0.8, 0.2],
            generator=torch.Generator().manual_seed(42),
        )
        # When computing the quantile labels, we use the test_transform
        # Update the val_ds transform  to remove the random crop
        train_ds.transform = test_transform
        val_ds.transform = test_transform
        self.train_ds = QuantileDataset(
            self.model,
            train_ds,
            quant_model_name=f"{self.name_base_model}_train",
            train=True,
            random_quantile=False,
            force_compute_cache=True,
        )
        self.val_ds = QuantileDataset(
            self.model,
            val_ds,
            quant_model_name=f"{self.name_base_model}_val",
            train=True,
            force_compute_cache=True,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            total_ds, _, _, test_transform = get_base_datasets(self.name_base_model)
            # Randomly permute the dataset
            train_ds, val_ds = random_split(
                total_ds,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
            # Update the val_ds transform  to remove the random crop
            val_ds.transform = test_transform
            self.train_ds = QuantileDataset(
                self.model,
                train_ds,
                quant_model_name=f"{self.name_base_model}_train",
                train=True,
                random_quantile=False,
                force_compute_cache=False,
            )
            self.val_ds = QuantileDataset(
                self.model,
                val_ds,
                quant_model_name=f"{self.name_base_model}_val",
                train=True,
                force_compute_cache=False,
            )

        if stage == "test" or stage is None:
            _, test_ds, _, _ = get_base_datasets(self.name_base_model)
            self.test_ds = QuantileDataset(
                self.model,
                test_ds,
                quant_model_name=f"{self.name_base_model}_test",
                train=False,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=14,
            drop_last=False,
            collate_fn=manual_collate_fn,
        )
        return dataloader

    def val_dataloader(self):
        dataLoader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=14,
            drop_last=False,
            collate_fn=manual_collate_fn,
        )
        return dataLoader

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            drop_last=False,
            collate_fn=manual_collate_fn,
        )
