#!/usr/bin/env python3
""" A training program for MNASNet 1.0.

All parameters are hardcoded for better reproducibility. This training harness
targets MNASNet with a depth multiplier of 1.0. By running it with these
parameters on a machine with 4 1080ti GPUs you should obtain approximately
73.512% top-1 accuracy on ImageNet. """

import os
import typing
import multiprocessing

import torch
import torchvision.models as models

import trainer
import imagenet
import metrics
import imagenet
import log
import tensorboard

NUM_EPOCHS = 200
BASE_LR = 0.7
MOMENTUM = 0.9
WARMUP = 5
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 740
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


def train() -> None:
    model = models.mnasnet1_0(1000).cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                                nesterov=True)
    loss = torch.nn.CrossEntropyLoss().cuda()
    lr_schedule = CosineWithWarmup(optimizer, WARMUP, 0.1, NUM_EPOCHS)

    train_dataset = imagenet.training(IMAGENET_DIR)
    val_dataset = imagenet.validation(IMAGENET_DIR)

    train = trainer.Trainer(
        ".", "MNASNet 1.0, cosine annealing with warmup, "
        "base_lr=0.7, 200 epochs.", "multiclass_classification", True, model,
        optimizer, loss, lr_schedule, metrics.default(), cudnn_autotune=True)

    train.fit(train_dataset, val_dataset, num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              num_workers=multiprocessing.cpu_count() // 2)


if __name__ == "__main__":
    log.open_log_file(output_dir=".")
    tensorboard.open_log_file(output_dir=".")
    try:
        train()
    finally:
        log.close_log_file()
        tensorboard.close_log_file()
