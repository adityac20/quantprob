import numpy as np
from PIL import Image
from glob import glob
from os import path as osp

import pdb
import torch

import torchvision
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader

import src.config as config
from src.models.densenet import DenseNet3
from src.models.resnet import ResNet34

import sys
from pathlib import Path


def get_pretrained_model(name_model, use_pretrained=True):
    """ """
    assert name_model in config.PRETRAINED_MODELS, f"Model {name_model} not found."
    if name_model == "densenet_cifar10":
        model = DenseNet3(100, 10)
        if use_pretrained:
            # model.load(config.PRETRAINED_DIR + "densenet_cifar10.pth")
            model.load_state_dict(
                torch.load(config.PRETRAINED_DIR + "densenet_cifar10_state_dict.pth")
            )
        num_classes = 10
        size_dataset = int(0.8 * 50000)
    elif name_model == "densenet_cifar100":
        model = DenseNet3(100, 100)
        if use_pretrained:
            model.load(config.PRETRAINED_DIR + "densenet_cifar100.pth")
        num_classes = 100
        size_dataset = int(0.8 * 50000)
    elif name_model == "densenet_svhn":
        model = DenseNet3(100, 10)
        if use_pretrained:
            model.load(config.PRETRAINED_DIR + "densenet_svhn.pth")
        num_classes = 10
        size_dataset = int(0.8 * 73257)
    elif name_model == "resnet34_cifar10":
        model = ResNet34(10)
        if use_pretrained:
            model.load_state_dict(
                torch.load(config.PRETRAINED_DIR + "resnet34_cifar10.pth")
            )
        num_classes = 10
        size_dataset = int(0.8 * 50000)
    elif name_model == "resnet34_cifar100":
        model = ResNet34(100)
        if use_pretrained:
            model.load_state_dict(
                torch.load(config.PRETRAINED_DIR + "resnet34_cifar100.pth")
            )
        num_classes = 100
        size_dataset = int(0.8 * 50000)
    elif name_model == "resnet34_svhn":
        model = ResNet34(10)
        if use_pretrained:
            model.load_state_dict(
                torch.load(config.PRETRAINED_DIR + "resnet34_svhn.pth")
            )
        num_classes = 10
        size_dataset = int(0.8 * 73257)

    return model, num_classes, size_dataset


def get_base_datasets(name_base_model):
    """Returns the Train and Test Datasets along with
    transforms for the input data.

    - Transforms are such that each channel is has mean 0 and std 1.
    """
    assert (
        name_base_model in config.PRETRAINED_MODELS
    ), f"Model {name_base_model} not found."
    if name_base_model == "densenet_cifar10" or name_base_model == "resnet34_cifar10":
        train_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914009, 0.48215896, 0.4465308),
                    (0.24703279, 0.24348423, 0.26158753),
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914009, 0.48215896, 0.4465308),
                    (0.24703279, 0.24348423, 0.26158753),
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=config.DATA_DIR, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=config.DATA_DIR, train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif (
        name_base_model == "densenet_cifar100" or name_base_model == "resnet34_cifar100"
    ):
        train_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070754, 0.48655024, 0.44091907),
                    (0.26733398, 0.25643876, 0.2761503),
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070754, 0.48655024, 0.44091907),
                    (0.26733398, 0.25643876, 0.2761503),
                ),
            ]
        )
        train_dataset = datasets.CIFAR100(
            root=config.DATA_DIR, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=config.DATA_DIR, train=False, download=True, transform=test_transform
        )
        num_classes = 100
    elif name_base_model == "densenet_svhn" or name_base_model == "resnet34_svhn":
        train_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070754, 0.48655024, 0.44091907),
                    (0.26733398, 0.25643876, 0.2761503),
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070754, 0.48655024, 0.44091907),
                    (0.26733398, 0.25643876, 0.2761503),
                ),
            ]
        )
        train_dataset = datasets.SVHN(
            root=config.DATA_DIR,
            split="train",
            download=True,
            transform=train_transform,
        )
        test_dataset = datasets.SVHN(
            root=config.DATA_DIR, split="test", download=True, transform=test_transform
        )
        num_classes = 10

    return train_dataset, test_dataset, num_classes, test_transform


def get_datatransform_ood_dataset(root):
    """
    - - Transforms are such that each channel is has mean 0 and std 1.
    """
    if "LSUN" in root:
        data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5107861, 0.47190297, 0.43479815),
                    (0.2737108, 0.27348495, 0.2900767),
                ),
            ]
        )
    elif "iSUN" in root:
        data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4861, 0.4633, 0.4275),
                    (0.2598, 0.2576, 0.2753),
                ),
            ]
        )
    elif "LSUN_resize" in root:
        data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5076, 0.4702, 0.4350),
                    (0.2745, 0.2760, 0.2896),
                ),
            ]
        )
    elif "Imagenet" in root:
        data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.47640708, 0.4364044, 0.38390368),
                    (0.2706367, 0.26130518, 0.27079648),
                ),
            ]
        )
    elif "Imagenet_resize" in root:
        data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4715, 0.4413, 0.3931),
                    (0.2738, 0.2674, 0.2775),
                ),
            ]
        )

    return data_transform


class ImageFolderOOD(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        img_path_list = (
            glob(osp.join(root, "*", "*.jpeg"))
            + glob(osp.join(root, "*", "*.png"))
            + glob(osp.join(root, "*", "*.jpg"))
            + glob(osp.join(root, "*", "*", "*.JPEG"))
            + glob(osp.join(root, "*", "*.JPEG"))
        )
        if len(img_path_list) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.data_paths = img_path_list
        self.targets = [0] * len(img_path_list)
        self.data_dir = root
        if self.transform is None:
            self.transform = get_datatransform_ood_dataset(root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data_paths[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(default_loader(img_path))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_paths)

    @property
    def num_classes(self):
        return 2
