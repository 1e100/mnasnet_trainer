#!/usr/bin/env python3

import os.path
import typing

import torch.utils.data
import torchvision.transforms

_IMAGENET_DEFAULT_DIR = os.path.expanduser("~/datasets/imagenet")
_IMAGENET_MEANS = [0.485, 0.456, 0.406]
_IMAGENET_STDS = [0.229, 0.224, 0.225]


def imagenet_normalize() -> torchvision.transforms.Normalize:
    return torchvision.transforms.Normalize(mean=_IMAGENET_MEANS,
                                            std=_IMAGENET_STDS, inplace=True)


def training_augmentation(size: int = 224) -> torchvision.transforms.Compose:
    """ Imagenet training augmentation stack, similar to what's in the paper
    (resize, crop, flip), but also includes ColorJitter that varies the
    brightness, contrast, saturation, and hue. """
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        imagenet_normalize(),
    ])


def validation_augmentation(size: int = 224) -> torchvision.transforms.Compose:
    """ `size` is the size of the final crop. Resize will be done to the next
    power of 2 size. I.e. for 224 resize is to 256. """
    resize_size = 2**int(size).bit_length()
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize_size),
        torchvision.transforms.CenterCrop(size),
        torchvision.transforms.ToTensor(),
        imagenet_normalize(),
    ])


def training(dataset_dir: str = _IMAGENET_DEFAULT_DIR,
             size: int = 224) -> torch.utils.data.Dataset:
    """ Training subset of ImageNet with default augmentation stack. """
    return torchvision.datasets.ImageFolder(
        os.path.join(dataset_dir, "train"),
        transform=training_augmentation(size=size))


def validation(dataset_dir: str = _IMAGENET_DEFAULT_DIR,
               size: int = 224) -> torch.utils.data.Dataset:
    """ Validation subset of ImageNet with default augmentation stack. """
    return torchvision.datasets.ImageFolder(
        os.path.join(dataset_dir, "val"),
        transform=validation_augmentation(size=size))
