#!/usr/bin/env python3
""" A training program for MNASNet family of models.

All parameters are hardcoded for better reproducibility. This training harness
targets MNASNet with various depth multipliers. To change the depth multiplier,
simply set `MODEL_NAME` variable to the model you want. 4x1080ti are assumed.
Fewer GPUs will require smaller batches and smaller learning rates."""

import os
import typing
import multiprocessing
import math

import torch
import torchvision.models as models

import trainer
import imagenet
import metrics
import imagenet
import log
import tensorboard

MODEL_NAME = "mnasnet0_75"
TRAINING_PARAMS = {
    "mnasnet0_5": {
        "num_epochs": 250,
        "base_lr": 1.0,
        "momentum": 0.9,
        "weight_decay": 0,
        "batch_size": 1000,
    },
    "mnasnet0_75": {
        "num_epochs": 300,
        "base_lr": 0.8,
        "momentum": 0.9,
        "weight_decay": 0,
        "batch_size": 1000,
    },
    "mnasnet1_0": {
        "num_epochs": 300,
        "base_lr": 0.7,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "batch_size": 740,
    },
    "mnasnet1_3": {
        "num_epochs": 300,
        "base_lr": 0.35,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "batch_size": 740,
    }
}

WARMUP = 5
IMAGENET_DIR = os.path.expanduser("~/datasets/imagenet")


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int,
                 last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = 0.0  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]


def train(model_name: str) -> None:
    if model_name == "mnasnet0_5":
        model = models.mnasnet0_5(1000).cuda()
    if model_name == "mnasnet0_75":
        model = models.mnasnet0_75(1000).cuda()
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(1000).cuda()
    if model_name == "mnasnet1_3":
        model = models.mnasnet1_3(1000).cuda()
    else:
        raise ValueError("Don't know how to train {}".format(model_name))
    params = TRAINING_PARAMS[model_name]

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=params["base_lr"], momentum=params["momentum"],
        weight_decay=params["weight_decay"], nesterov=True)
    loss = torch.nn.CrossEntropyLoss().cuda()
    lr_schedule = CosineWithWarmup(optimizer, WARMUP, 0.1, params["num_epochs"])

    train_dataset = imagenet.training(IMAGENET_DIR)
    val_dataset = imagenet.validation(IMAGENET_DIR)

    train = trainer.Trainer(
        ".", "MNASNet 0.5, cosine annealing with warmup, "
        "base_lr=1.0, 250 epochs.", "multiclass_classification", True, model,
        optimizer, loss, lr_schedule, metrics.default(), cudnn_autotune=True)

    train.fit(train_dataset, val_dataset, num_epochs=params["num_epochs"],
              batch_size=params["batch_size"],
              num_workers=multiprocessing.cpu_count())


if __name__ == "__main__":
    log.open_log_file(output_dir=".")
    tensorboard.open_log_file(output_dir=".")
    try:
        train(MODEL_NAME)
    finally:
        log.close_log_file()
        tensorboard.close_log_file()
