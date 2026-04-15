import logging
import threading

try:
    import logging_mp as _logging_mp
except ImportError:
    _logging_mp = None

_CONFIG_LOCK = threading.Lock()
_CONFIGURED = False


def _configure_logging(level: int | None) -> None:
    global _CONFIGURED

    if _CONFIGURED or _logging_mp is None:
        return

    with _CONFIG_LOCK:
        if _CONFIGURED:
            return

        config_level = level if level is not None else logging.WARNING
        if hasattr(_logging_mp, "basicConfig"):
            _logging_mp.basicConfig(level=config_level)
        elif hasattr(_logging_mp, "basic_config"):
            _logging_mp.basic_config(level=config_level)

        _CONFIGURED = True


def _get_fallback_logger(name: str | None, level: int | None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str | None = None, level: int | None = None) -> logging.Logger:
    if _logging_mp is None:
        return _get_fallback_logger(name, level)

    _configure_logging(level)

    if hasattr(_logging_mp, "getLogger"):
        logger = _logging_mp.getLogger(name)
    elif hasattr(_logging_mp, "get_logger"):
        logger = _logging_mp.get_logger(name, level=level)
    else:
        return _get_fallback_logger(name, level)

    if level is not None:
        logger.setLevel(level)

    return logger
