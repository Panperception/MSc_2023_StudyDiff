import logging
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger("deep_learning_logger")
logger.setLevel(logging.DEBUG)

file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

ch = TimedRotatingFileHandler(
    "/root/autodl-tmp/app_logs/deep_learning.log", when="midnight", interval=1
)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
file_logger.addHandler(ch)