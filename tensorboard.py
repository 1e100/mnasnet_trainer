#!/usr/bin/env python3

import os
import datetime
import socket

import tensorboardX


def _get_tensorboard_subdir():
    timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", timestamp + "_" + socket.gethostname())


TENSORBOARD_SUBDIR = _get_tensorboard_subdir()

_writer = None


def open_log_file(output_dir="/tmp"):
    """ Initializes Tensorboard log writer. If it is not initialized, `add_`
    methods will not do anything. """
    global _writer
    # Tensorboard logs into a timestamped subdir of `output_dir` under `runs/`
    tensorboard_dir = os.path.join(output_dir, TENSORBOARD_SUBDIR)
    _writer = tensorboardX.SummaryWriter(log_dir=tensorboard_dir)


def close_log_file():
    global _writer
    if _writer:
        _writer.close()
        _writer = None


def add_scalar(tag, scalar_value, global_step=None, walltime=None):
    """ Adds a scalar value for a given tag, e.g. `learning_rate` or `train_loss` """
    if _writer:
        _writer.add_scalar(tag, scalar_value, global_step=global_step,
                           walltime=walltime)


def add_scalars(main_tag, tag_scalar_dict, global_step=None):
    """ Adds several scalar values from dict under a main_tag, e.g. `{"top5":
        acc_top_5, "loss": val_loss}` under `validation`. """
    if _writer:
        _writer.add_scalars(main_tag, tag_scalar_dict, global_step=global_step)


def add_image(tag, image, global_step=None, walltime=None):
    """ Adds an image (tensor or ndarray) for a given tag. """
    if _writer:
        _writer.add_image(tag, image, global_step=global_step,
                          walltime=walltime)


def add_images(tag, images, global_step=None):
    """ Adds several images (tensor, ndarray or string) arranged in a grid. """
    if _writer:
        _writer.add_images(tag, images, global_step=global_step)


def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None):
    """ Adds image with bounding boxes overlaid. For detection. """
    if _writer:
        _writer.add_image_with_boxes(tag, img_tensor, box_tensor,
                                     global_step=global_step)


def add_text(tag, text, global_step=None):
    if _writer:
        _writer.add_text(tag, text, global_step=global_step)


def add_figure(tag, figure, global_step=None, walltime=None):
    """ Adds a matplotlib figure for a given tag. """
    if _writer:
        _writer.add_figure(tag, figure, global_step=global_step,
                           walltime=walltime)
