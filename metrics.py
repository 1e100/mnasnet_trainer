#!/usr/bin/env python3

from typing import List, Callable

import torch


class TopKMetric:

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, output: torch.Tensor, truth: torch.Tensor) -> float:
        """ Computes the precision@k for the specified value of k. """
        with torch.no_grad():
            batch_size = truth.size(0)
            _, pred = output.topk(self.k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(truth.view(1, -1).expand_as(pred))
            correct_k = correct[:self.k].view(-1).float().sum(0, keepdim=True)
            return correct_k.mul_(100.0 / batch_size)


def default() -> List[Callable[[torch.Tensor, torch.Tensor], float]]:
    return [("prec1", TopKMetric(1)), ("prec5", TopKMetric(5))]
