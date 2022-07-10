#!/usr/bin/env python3

import re
import logging
import coloredlogs


class Logger(object):
    """
    Wrapper over coloredlogs package that enables colored terminal output for Python logging module
    """

    log_format = '[%(asctime)s] (%(threadName)s) {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    log_level = None

    @classmethod
    def setup_logging(cls, loglevel='INFO', logfilter=".*", logfile=""):
        cls.registered_loggers = dict()
        cls.log_level = loglevel
        cls.log_logfilter_regexp = logfilter
        cls.log_logfilter_regexp_compiled = re.compile(logfilter)

        numeric_level = getattr(logging, loglevel.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        if logfile:
            logging.basicConfig(handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
                                level=numeric_level,
                                format=cls.log_format,
                                datefmt='%Y-%m-%d %H:%M:%S',)
        else:
            logging.basicConfig(level=numeric_level,
                                format=cls.log_format,
                                datefmt='%Y-%m-%d %H:%M:%S',)

    @classmethod
    def get(cls, logger_name='default'):
        if logger_name in cls.registered_loggers:
            return cls.registered_loggers[logger_name]
        else:
            return cls(logger_name)

    def __init__(self, logger_name='default'):
        """
        Instantiate logger

        Args:
            logger_name(str): logger name
        """
        if logger_name in self.registered_loggers:
            raise ValueError(f"Logger {logger_name} already exists. Call with Logger.get(\"{logger_name}\")")
        else:
            self.name = logger_name
            self.logger = logging.getLogger(self.name)

            # FIX: instead of registering system logger - register self with custom filtering logic
            # INCORRECT: self.registered_loggers[self.name] = self.logger
            self.registered_loggers[self.name] = self

            coloredlogs.install(
                level=self.log_level,
                logger=self.logger,
                fmt=self.log_format,
                datefmt='%Y-%m-%d %H:%M:%S')

    def log(self, loglevel, msg):
        """
        Main entry point for log text messages.
        The logging module is intended to be thread-safe without any special work needing to be done by its clients
           https://docs.python.org/3/library/logging.html#thread-safety

        Args:
            loglevel(str): logging level is one of case insensitive string: DEBUG, INFO, WARNING, ERROR, CRITICAL
            msg(str): text message to log
        """
        if not self.log_logfilter_regexp_compiled.match(msg):
            return

        loglevel = loglevel.upper()
        if loglevel == 'DEBUG':
            self.logger.debug(msg)
        elif loglevel == 'INFO':
            self.logger.info(msg)
        elif loglevel == 'WARNING':
            self.logger.warning(msg)
        elif loglevel == 'ERROR':
            self.logger.error(msg)
        elif loglevel == 'CRITICAL':
            self.logger.critical(msg)

    def debug(self, msg):
        """
        Log message in debug mode.

        Args:
            msg(str): text message to log
        """
        self.log('debug', msg)

    def info(self, msg):
        """
        Log message in info mode.

        Args:
            msg(str): text message to log
        """
        self.log('info', msg)

    def warning(self, msg):
        """
        Log message in warning mode.

        Args:
            msg(str): text message to log
        """
        self.log('warning', msg)

    def error(self, msg):
        """
        Log message in error mode.

        Args:
            msg(str): text message to log
        """
        self.log('error', msg)

    def critical(self, msg):
        """
        Log message in critical mode.

        Args:
            msg(str): text message to log
        """
        self.log('critical', msg)
