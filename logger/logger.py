import logging
import logging.config
import os
import yaml


def setup_logging(save_path='logs', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(message)s"
            },
            "datetime": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "datetime",
                "filename": "info.log",
                "maxBytes": 10485760,
                "backupCount": 20,
                "encoding": "utf8"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "info_file_handler"]
        },
    }

    # print(config)

    for _, handler in config["handlers"].items():
        print(handler)
        if "filename" in handler:
            handler["filename"] = os.path.join(save_path, handler["filename"])

    logging.config.dictConfig(config)

    return logging.getLogger(__name__)


def TensorboardWriter(log_dir, logger, enabled=True):
    """
    Tensorboard writer
    """
    raise NotImplementedError


if __name__ == "__main__":
    logger = setup_logging("logs")
    
    # logger.info("Hello, world!")
