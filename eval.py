#!/usr/bin/env python3
""" An evaluation script for MNASNet. """

import os
import typing
import collections

import torch
import torchvision.models as models
import numpy
import tqdm

import imagenet
import metrics
import log

IMAGENET_DIR = os.path.expanduser("~/datasets/imagenet")


def eval(model_name: str) -> None:
    if model_name == "mnasnet0_5":
        model = models.mnasnet0_5(num_classes=1000, pretrained=True).cuda()
    elif model_name == "mnasnet0_75":
        model = models.mnasnet0_75(num_classes=1000).cuda()
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(num_classes=1000, pretrained=True).cuda()
    elif model_name == "mnasnet1_3":
        model = models.mnasnet1_3(num_classes=1000).cuda()
    else:
        raise ValueError("Don't know how to evaluate {}".format(model_name))

    loss = torch.nn.CrossEntropyLoss().cuda()
    val_dataset = imagenet.validation(IMAGENET_DIR)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512,
                                             shuffle=False, num_workers=8,
                                             pin_memory=True)
    all_metrics = metrics.default()

    model.eval()
    with torch.no_grad():
        val_losses = []
        metric_dict = collections.defaultdict(list)
        for batch_index, (inputs, truth) in enumerate(tqdm.tqdm(val_loader)):
            outputs = model(inputs.cuda()).cpu()
            val_losses.append(loss(outputs, truth).item())
            for name, metric_fn in all_metrics:
                metric_dict[name].append(metric_fn(outputs, truth))

        print(
            numpy.mean(val_losses),
            list([(name, numpy.mean(vals))
                  for name, vals in metric_dict.items()]))


if __name__ == "__main__":
    for m in ["mnasnet1_0", "mnasnet0_5"]:
        print("Evaluating pretrained", m)
        eval(m)
