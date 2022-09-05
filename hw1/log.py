import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

LOGGER_NAME = 'athang213'
LOG_FILE_NAME = ''

def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


def setup_logging():
    filepath = next_path("logs/run-%s.log")
    global LOG_FILE_NAME
    LOG_FILE_NAME = filepath.split('/')[-1]

    log = logging.getLogger(LOGGER_NAME)
    formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S'
        )
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    fileHandler = RotatingFileHandler(filename=filepath, maxBytes=2*1024*1024, backupCount=5)
    fileHandler.setFormatter(formatter)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)
    log.setLevel(logging.DEBUG)

    log.info("Logger configured successfully")

def get_logger():
    return logging.getLogger(LOGGER_NAME)