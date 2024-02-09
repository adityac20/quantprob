import logging
from glob import glob
from os import path as osp
from os.path import join as osj

import numpy as np
import torch

from PIL import Image
from torchvision import datasets
from torchvision import transforms


# distortions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise',
#     'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#     'snow', 'frost', 'fog', 'brightness',
#     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
#     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
# ]

distortions_paper = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def get_distorted_imagenet(distortion_name, severity):
    """Return the distorted dataset"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    path_name = f"./data/ImagenetC/{distortion_name}/{severity}"
    transform_ImagenetC = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    distorted_dataset = datasets.ImageFolder(
        root=path_name, transform=transform_ImagenetC
    )
    return distorted_dataset


class DistortedCIFAR10(datasets.VisionDataset):
    """
    In CIFAR-10-C, the first 10,000 images in each .npy are the test set images
    corrupted at severity 1, and the last 10,000 images are the test set images
    corrupted at severity five. labels.npy is the label file for all other image
    files.

    @article{hendrycks2019robustness,
      title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
      author={Hendrycks, Dan and Dietterich, Thomas},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2019}
    }

    """

    list_distorions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    list_severity = [1, 2, 3, 4, 5]

    def __init__(
        self, root, distortion, severity, transform, target_transform=None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = np.load(f"{root}/CIFAR10C/CIFAR-10-C/{distortion}.npy")
        self.targets = np.load(f"{root}/CIFAR10C/CIFAR-10-C/labels.npy")
        self.ind_start, self.ind_end = (severity - 1) * 10000, (severity) * 10000
        self.data_dir = root
        self.num_classes = 10

    def __getitem__(self, index):
        img, target = self.data[self.ind_start + index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return 10000
