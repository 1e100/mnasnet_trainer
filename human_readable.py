#!/usr/bin/env python3

import typing


def duration(elapsed_sec: float) -> str:
    """ Converts duration into a human readable format.

    >>> duration(1.1)
    '1.100'
    >>> duration(61.15)
    '1:01.150'
    >>> duration(3601.123)
    '1:00:01.123'
    """
    hours = (int(elapsed_sec) // 3600) % 24
    minutes = (int(elapsed_sec) // 60) % 60
    seconds = int(elapsed_sec) % 60
    ms = int(elapsed_sec * 1000) % 1000
    if hours > 0:
        return "{:d}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, ms)
    elif minutes > 0:
        return "{:d}:{:02d}.{:03d}".format(minutes, seconds, ms)
    else:
        return "{:d}.{:03d}".format(seconds, ms)
