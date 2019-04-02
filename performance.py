#!/usr/bin/env python3

import numpy as np
import time


class MovingAverage:
    """ Computes a moving average of a set of floating point values.

    >>> m = MovingAverage(32)
    >>> m.get()
    0.0
    >>> m.add(1.0)
    >>> m.get()
    1.0
    >>> m.add(2.0)
    >>> m.get()
    1.5
    >>> for x in range(100): m.add(x)
    >>> m.get()
    83.5
    """

    def __init__(self, window_size: int) -> None:
        self.window = np.zeros(window_size, dtype=np.float)
        self.index = 0

    def add(self, val: float) -> None:
        self.window[self.index % len(self.window)] = val
        self.index += 1

    def get(self) -> float:
        if self.index == 0:
            return 0.0
        return np.mean(self.window[:min(self.index, len(self.window))])


class Timer:
    """ A timer context manager that updates a given moving average.

    >>> m = MovingAverage(8)
    >>> with Timer(m): time.sleep(0.05)
    >>> 0.04 < m.get() < 0.06
    True
    >>> with Timer(m): time.sleep(0.1)
    >>> 0.06 < m.get() < 0.08
    True
    """

    def __init__(self, moving_average: MovingAverage):
        self.moving_average = moving_average

    def __enter__(self) -> None:
        self.start_time = time.perf_counter()

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.moving_average.add(elapsed)
