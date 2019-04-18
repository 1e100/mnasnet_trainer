from typing import Tuple, List, Callable, Dict, Any
import collections
import itertools
import json
import math
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import re
import datetime

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import tensorboard
import log
import performance
import human_readable

LOG_INTERVAL = 10
CONFIG_FILE_NAME = "run.config"
CHECKPOINT_FILE_NAME = "checkpoint.pth"
BEST_CHECKPOINT_FILE_NAME = "best_" + CHECKPOINT_FILE_NAME


def _full_type_name(obj):
    return "{}.{}".format(obj.__module__, type(obj).__qualname__)


class Trainer:

    def __init__(self, run_dir: str, run_description: str, problem_type: str,
                 load_checkpoint: bool, model: nn.Module,
                 optimizer: optim.Optimizer, loss: nn.Module,
                 lr_scheduler: optim.lr_scheduler._LRScheduler, metrics: List[
                     Tuple[str, Callable[[torch.Tensor, torch.Tensor], Any]]],
                 cudnn_autotune: bool):
        # Enable autotuning in CUDNN. Note that it is not a good idea to enable
        # this when input sizes or network structure may be different iteration
        # to iteration.
        torch.backends.cudnn.benchmark = cudnn_autotune
        run_dir = os.path.expanduser(run_dir)
        assert os.path.isdir(run_dir)
        self.run_dir = run_dir
        self.devices = [
            torch.device("cuda:{}".format(d))
            for d in range(torch.cuda.device_count())
        ]
        self.model = model
        if len(self.devices) > 1 and not isinstance(model, nn.DataParallel):
            log.fatal(
                "{} GPUs found, but model is not DataParallel. "
                "Wrap it into DataParallel for greater throughput.".format(
                    torch.cuda.device_count()))
        self.optimizer = optimizer
        self.loss_function = loss
        self.lr_scheduler = lr_scheduler  # May be None
        self.metrics = metrics
        self.epoch = 0
        self.global_step = 0
        self.best_loss = sys.float_info.max
        if problem_type not in [
                "multiclass_classification", "multilabel_classification"
        ]:
            raise ValueError(
                "Unsupported problem type: {}".format(problem_type))
        self.problem_type = problem_type  # E.g. "multilabel_classification"
        self.run_description = run_description

        self.checkpoint_path = os.path.join(self.run_dir, CHECKPOINT_FILE_NAME)
        self.best_checkpoint_path = os.path.join(self.run_dir,
                                                 BEST_CHECKPOINT_FILE_NAME)
        self.average_data_duration = performance.MovingAverage(32)
        self.average_compute_duration = performance.MovingAverage(32)
        # Checkpoint reload, if requested.
        if load_checkpoint:
            self._load_checkpoint()

    def fit(self, train_set: torch.utils.data.Dataset,
            validation_set: torch.utils.data.Dataset, num_epochs: int = 100,
            batch_size: int = 128,
            num_workers: int = multiprocessing.cpu_count()):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            validation_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        log.info(">> Starting training at epoch {}, global step {}.".format(
            self.epoch, self.global_step))
        # If data parallel model, obtain the type name of the wrapped
        # model.
        model_type_name = _full_type_name(self._unwrap_parallel_model())
        log.info(">> Model class: {}", model_type_name)
        log.info(">> Optimizer: {}", _full_type_name(self.optimizer))
        log.info(">> Loss function: {}", _full_type_name(self.loss_function))
        if self.lr_scheduler:
            log.info(">> Learning rate scheduler: {}",
                     _full_type_name(self.lr_scheduler))
        log.info(">> Batch size: {}", batch_size)
        log.info(">> Run description: {}", self.run_description)
        # This needs to go last because output is quite voluminous and we want
        # to be able to see short info blurb at the start of the log.
        log.info(">> Model definition:\n{}", self._unwrap_parallel_model())

        # If dataset has label_dict() function, obtain the label dict and write
        # it to JSON file in run_dir.
        get_dict_fn = getattr(train_set, "label_dict", None)
        if get_dict_fn and callable(get_dict_fn):
            label_dict = get_dict_fn()
            self._write_json(label_dict, "label_dict.json")

        # Save parameters that we might need when running a demo.
        params = {
            "model_type_name": model_type_name,
            "problem_type": self.problem_type,
            "description": self.run_description
        }
        self._write_json(params, "parameters.json")

        for self.epoch in range(self.epoch, num_epochs):
            # Learning rate.
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch=self.epoch)
                # Note that this may actually be a list, for different parameter
                # groups.
                lr = self.lr_scheduler.get_lr()[0]
                log.info("Learning rate is {}", lr)
                tensorboard.add_scalar("train/lr", lr,
                                       global_step=self.global_step)

            # Training
            self._train(train_loader)
            # Validation
            val_loss, val_metrics = self._validate(val_loader)

            # Log results, save checkpoint, save best checkpoint
            # if best model is found.
            tensorboard.add_scalar("val/loss", val_loss,
                                   global_step=self.global_step)
            for m in val_metrics:
                tensorboard.add_scalar("val/{}".format(m[0]), m[1],
                                       global_step=self.global_step)
            metrics_str = ", ".join(
                ["{}={:.3f}".format(m[0], m[1]) for m in val_metrics])
            log.info("VAL: loss={:.3f}, metrics = {}", val_loss, metrics_str)
            is_new_best = self.best_loss > val_loss
            if is_new_best:
                log.info("Found new best model, loss {:.5f} (was {}). "
                         "Saving checkpoint to {}.".format(
                             val_loss, "{:.5f}".format(self.best_loss)
                             if self.best_loss != sys.float_info.max else
                             "sys.float_info.max", self.best_checkpoint_path))
                self.best_loss = val_loss
            self._save_checkpoint(is_new_best)

    def _train(self, train_loader: torch.utils.data.DataLoader) -> None:
        log.info("Training epoch {}, global step {}", self.epoch,
                 self.global_step)
        start = time.time()
        self.model.train()
        losses = []
        # We can't wrap data load into a `Timer`, so this is a workaround. Note
        # that start time is also set at the very end of the batch loop.
        data_start_time = time.perf_counter()
        for batch_index, (inputs, annotations) in enumerate(train_loader, 0):
            compute_start_time = time.perf_counter()
            self.average_data_duration.add(compute_start_time - data_start_time)
            # DataParallel models handle device placement of inputs on their own.
            if not isinstance(self.model, nn.DataParallel):
                inputs = inputs.to(self.devices[0], non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, annotations.to(self.devices[0]))
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.average_compute_duration.add(time.perf_counter() -
                                              compute_start_time)
            if math.isnan(loss.item()):
                raise ValueError("Training blew up. Loss is NaN.")
            if batch_index % LOG_INTERVAL == 0:
                log.info(
                    "Epoch: {} [{}/{} ({:.0f}%), D:{:.2f}/C:{:.2f}s] "
                    "Loss: {:.5f}", self.epoch, batch_index * len(inputs),
                    len(train_loader.dataset),
                    100. * batch_index / len(train_loader),
                    self.average_data_duration.get(),
                    self.average_compute_duration.get(), loss.item())
                tensorboard.add_scalar("train/loss".format(self.epoch),
                                       loss.item(),
                                       global_step=self.global_step)
            self.global_step += 1
            # Console output is excluded from timings.
            data_start_time = time.perf_counter()
        mean_loss = numpy.mean(losses)
        tensorboard.add_scalar("train/mean_loss", mean_loss,
                               global_step=self.epoch)
        log.info("Mean training loss for epoch {}: {:.5f}", self.epoch,
                 mean_loss)

        training_elapsed_sec = time.time() - start
        log.info("Training duration: {}",
                 human_readable.duration(training_elapsed_sec))
        # Note: this relies on epoch covering the entire dataset.
        log.info("Training throughput: {:.2f} samples/sec".format(
            len(train_loader.dataset) / training_elapsed_sec))
        return self.global_step

    def _validate(self, val_loader: torch.utils.data.DataLoader
                 ) -> Tuple[numpy.float64, List[Tuple[str, float]]]:
        """ Returns validation lost and a list of (name, metric) tuples to print. """
        log.info("Validation epoch {}, global step {}", self.epoch,
                 self.global_step)
        start = time.time()
        self.model.eval()
        with torch.no_grad():
            val_losses = []
            metrics = collections.defaultdict(list)
            for batch_index, (inputs, truth) in enumerate(val_loader, 0):
                outputs = self.model(
                    inputs.to(self.devices[0], non_blocking=True)).cpu()
                val_losses.append(self.loss_function(outputs, truth).item())
                for name, metric_fn in self.metrics:
                    metrics[name].append(metric_fn(outputs, truth))

            log.info("Validation duration: {}",
                     human_readable.duration(time.time() - start))
            return numpy.mean(val_losses), list(
                [(name, numpy.mean(vals)) for name, vals in metrics.items()])

    def _unwrap_parallel_model(self) -> nn.Module:
        """ This "unwraps" DataParallel models such that they could then be
        saved/reloaded into a single-GPU model without name conflicts. """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def _load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            # Model is first loaded onto CPU and then moved to GPU if needed.
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            # If model was DataParallel, load the inner non-parallel model.
            model = self._unwrap_parallel_model()
            model_type = checkpoint["model_type"]
            if model_type != _full_type_name(model):
                raise ValueError(
                    "Checkpoint {} contains model {}, but "
                    "initialization provides model {}. Refusing to "
                    "load checkpoint.".format(self.checkpoint_path, model_type,
                                              _full_type_name(model)))
            model.load_state_dict(checkpoint["model_state_dict"])

            # Optimizer
            optimizer_type = checkpoint["optimizer_type"]
            if optimizer_type == _full_type_name(self.optimizer):
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
            else:
                log.info(
                    "Optimizer in checkpoint: {}, optimizer in init: "
                    "{}. Not loading optimizer state.", optimizer_type,
                    _full_type_name(self.optimizer))

            # Loss
            loss_type = checkpoint["loss_type"]
            if loss_type == _full_type_name(self.loss_function):
                self.best_loss = checkpoint["best_loss"]
            else:
                log.info(
                    "Loss in checkpoint: {}, loss in init: {}, "
                    "running best loss will be reset to sys.float_info.max.".
                    format(loss_type, _full_type_name(self.loss_function)))
                self.best_loss = sys.float_info.max

            self.epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"] + 1
        else:
            log.info("Checkpoint not found in {}, starting from scratch.",
                     self.run_dir)

    def _save_checkpoint(self, is_new_best: bool) -> None:
        """ Saves a training checkpoint. """
        # If model is parallel, get the non-parallel model from within.
        model = self._unwrap_parallel_model()

        checkpoint_state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss_type": _full_type_name(self.loss_function),
            "best_loss": self.best_loss,
            "model_type": _full_type_name(model),
            "model_state_dict": model.state_dict(),
            "optimizer_type": _full_type_name(self.optimizer),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler:
            checkpoint_state["lr_scheduler_type"] = _full_type_name(
                self.lr_scheduler)
            checkpoint_state[
                "lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint_state, self.checkpoint_path)
        # Save all model checkpoints with timestamp. A better way to do this
        # would be to track the metric and save checkpoints when metric
        # improves, but that's too much work for now.
        ts = re.sub("[:-]", "", datetime.datetime.now().isoformat())
        torch.save(model.state_dict(), "{}-{}".format(self.best_checkpoint_path, ts))
        # If a new best model was found, set aside a "bare" model checkpoint for it.
        if is_new_best:
            torch.save(model.state_dict(), self.best_checkpoint_path)

    def _write_json(self, dict: Dict[str, Any], file_name: str) -> None:
        """ Writes `dict` as JSON file to the current `run_dir`. """
        with open(os.path.join(self.run_dir, file_name), "w") as out_f:
            json.dump(dict, out_f, sort_keys=True, indent=2, ensure_ascii=False)
