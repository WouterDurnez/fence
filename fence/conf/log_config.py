from fence.utils.logger import ColorFormatter

config_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        },
        "color": {
            "()": ColorFormatter
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "color",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

