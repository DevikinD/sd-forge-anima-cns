import logging

from backend.logging import setup_logger

LOGGER = logging.getLogger("colored_noise")
setup_logger(LOGGER)

_TAG = "[ColoredNoise]"
_seen_once = set()


def info(msg, *args):
    LOGGER.info(_TAG + " " + msg, *args)


def warning(msg, *args):
    LOGGER.warning(_TAG + " " + msg, *args)


def debug(msg, *args):
    LOGGER.debug(_TAG + " " + msg, *args)


def debug_enabled():
    return LOGGER.isEnabledFor(logging.DEBUG)


def warn_once(key, msg, *args):
    """Emit a WARNING only the first time a given key is seen (avoids per-step spam)."""
    if key in _seen_once:
        return
    _seen_once.add(key)
    LOGGER.warning(_TAG + " " + msg, *args)
