#!/usr/bin/env python3

import os
import sys
import datetime
import traceback

DEBUG = 5
INFO = 4
WARNING = 3
ERROR = 2
FATAL = 1
_FATAL_NOTHROW = 0  # Same as fatal, but doesn't throw.

LEVEL_MAP = {
    DEBUG: "D",
    INFO: "I",
    WARNING: "W",
    ERROR: "E",
    FATAL: "F",
    _FATAL_NOTHROW: "F"
}

DEFAULT_LOG_DIR = "/tmp"
DEFAULT_LOG_FILE = os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".log"


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]


START_TIMESTAMP = get_timestamp()


class FatalError(RuntimeError):

    def __init__(self, message):
        super().__init__(message)


class Logger:

    def __init__(self, level=INFO):
        """ It is expected that `output_dir` already exists. """
        self._file = None
        self._level = level

    def _write(self, level, fmt, *args):
        if level > self._level:
            return
        message = fmt.format(*args)
        level_abbrev = LEVEL_MAP[level]
        stream = sys.stdout if level > ERROR else sys.stderr
        timestamp = datetime.datetime.now().strftime("%m%d %H%M%S.%f")[:-3]
        prefix = "{}{} {}".format(level_abbrev, timestamp, os.getpid())
        console_message = "{}] {}".format(prefix, message)
        print(console_message, file=stream)
        if self._file:
            # Log on disk will also contain the call site.
            caller = traceback.extract_stack(limit=3)[0]
            caller_file = os.path.basename(caller.filename)
            file_message = "{} {}:{}] {}".format(prefix, caller_file,
                                                 caller.lineno, message)
            print(file_message, file=self._file)
        if level == FATAL:
            raise FatalError(message)

    def open_file(self, output_dir, log_file_name):
        """ Opens the log file. Unless this is called, all logging will only be
        done to the console. """
        assert self._file == None
        assert os.path.isdir(output_dir)
        log_path = os.path.join(output_dir, log_file_name)
        self._file = open(log_path, "a", 1)  # Log will be line-buffered.

    def close_file(self):
        """ Closes the log file and stops logging to disk. """
        if self._file:
            self._file.close()
            self._file = None


# By default, this will only write to console. If you'd like to log into a file
# as well, call `open_file()`.
_logger = Logger(level=INFO)
_tensorboard_writer = None


def set_level(level):
    if level not in LEVEL_MAP or level == _FATAL_NOTHROW:
        raise ValueError("Log level may not be set to {}.".format(level))


def open_log_file(output_dir=DEFAULT_LOG_DIR, log_file_name=DEFAULT_LOG_FILE):
    """ Opens the log file. All output to log methods will be duplicated to the
    provided log file. Default filename is the name of the program with the
    extension stripped, followed by '.log'. """
    _logger.open_file(output_dir=output_dir, log_file_name=log_file_name)


def close_log_file():
    """ Closes the log file. """
    _logger.close_file()


def debug(fmt, *args):
    _logger._write(DEBUG, fmt, *args)


def info(fmt, *args):
    _logger._write(INFO, fmt, *args)


def warning(fmt, *args):
    _logger._write(WARNING, fmt, *args)


def error(fmt, *args):
    _logger._write(ERROR, fmt, *args)


def fatal(fmt, *args):
    """ Raises FatalError, a subclass of RuntimeError. """
    _logger._write(FATAL, fmt, *args)


# TODO: Filter stack frames within this module, if possible.
def exception(exception):
    """ Logs exception with traceback, if traceback is present. """
    assert issubclass(exception.__class__, Exception)
    if exception.__traceback__:
        frames = traceback.format_tb(exception.__traceback__)
        message = "\n".join([f.rstrip() for f in frames])
        _logger._write(_FATAL_NOTHROW, "Traceback:")
        _logger._write(_FATAL_NOTHROW, message)
    _logger._write(_FATAL_NOTHROW, str(exception))
