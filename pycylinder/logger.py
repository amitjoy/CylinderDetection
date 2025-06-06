"""
A simple logger wrapper that provides consistent logging across the pycylinder package.
"""
import os
import sys
import time
from typing import Optional, Union, Callable, Dict, Any
from enum import Enum, auto


class LogLevel(Enum):
    """Log levels for controlling verbosity."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class CylinderLogger:
    """
    A configurable logger that can write to console, file, or both with different log levels.
    """
    def __init__(
        self, 
        mode: str = 'console', 
        log_file: Optional[str] = None,
        console_level: LogLevel = LogLevel.INFO,
        file_level: LogLevel = LogLevel.DEBUG,
        include_timestamp: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            mode: 'console', 'file', or 'both'
            log_file: Path to log file (required if mode is 'file' or 'both')
            console_level: Minimum log level for console output
            file_level: Minimum log level for file output
            include_timestamp: Whether to include timestamps in log messages
        """
        self.mode = mode
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level
        self.include_timestamp = include_timestamp
        
        # Validate mode and log_file
        if mode not in ['console', 'file', 'both']:
            raise ValueError("mode must be 'console', 'file', or 'both'")
            
        if mode in ['file', 'both']:
            if not log_file:
                raise ValueError("log_file must be provided when mode is 'file' or 'both'")
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:  # Only try to create directory if path is not empty
                os.makedirs(log_dir, exist_ok=True)
            # Clear the log file
            with open(log_file, 'w') as f:
                f.write('')  # Clear file at start
    
    def isEnabledFor(self, level: LogLevel) -> bool:
        """
        Check if logging is enabled for the specified level.
        Compatible with Python's standard logging interface.
        
        Args:
            level: LogLevel to check
            
        Returns:
            bool: True if logging is enabled for the specified level
        """
        # Check if either console or file logging is enabled for this level
        console_enabled = (self.mode in ['console', 'both']) and (level.value >= self.console_level.value)
        file_enabled = (self.mode in ['file', 'both']) and (level.value >= self.file_level.value)
        return console_enabled or file_enabled
    
    def _format_message(self, message: str, level: str) -> str:
        """Format the log message with timestamp and level."""
        timestamp = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " if self.include_timestamp else ""
        level_str = f"[{level.upper()}] "
        return f"{timestamp}{level_str}{message}"
    
    def _log_to_console(self, message: str, level: LogLevel) -> None:
        """Log a message to the console if level is sufficient."""
        if self.mode in ['console', 'both'] and level.value >= self.console_level.value:
            print(message, file=sys.stderr if level.value >= LogLevel.WARNING.value else sys.stdout)
    
    def _log_to_file(self, message: str, level: LogLevel) -> None:
        """Log a message to the file if level is sufficient."""
        if self.mode in ['file', 'both'] and self.log_file and level.value >= self.file_level.value:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
    
    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """
        Log a message with the specified level.
        
        Args:
            message: The message to log
            level: The log level (default: INFO)
        """
        formatted_msg = self._format_message(message, level.name.lower())
        self._log_to_console(formatted_msg, level)
        self._log_to_file(formatted_msg, level)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, LogLevel.INFO)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, LogLevel.WARNING)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, LogLevel.ERROR)
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.log(message, LogLevel.CRITICAL)
    
    def __call__(self, message: str, level: Union[str, LogLevel] = LogLevel.INFO) -> None:
        """
        Allow the logger instance to be called directly.
        
        Args:
            message: The message to log
            level: The log level as a string or LogLevel enum (default: 'info')
        """
        if isinstance(level, str):
            level = getattr(LogLevel, level.upper(), LogLevel.INFO)
        self.log(message, level)


# Default logger instance
DEFAULT_LOGGER = CylinderLogger(mode='console')


def get_logger(name: Optional[str] = None) -> CylinderLogger:
    """
    Get a logger instance. For API compatibility with Python's logging module.
    
    Args:
        name: Optional logger name (ignored, kept for compatibility)
    """
    return DEFAULT_LOGGER


def set_logger(logger: Union[CylinderLogger, None]) -> None:
    """
    Set the default logger instance.
    
    Args:
        logger: A CylinderLogger instance or None to reset to default
    """
    global DEFAULT_LOGGER
    if logger is None:
        DEFAULT_LOGGER = CylinderLogger(mode='console')
    elif not isinstance(logger, CylinderLogger):
        raise ValueError("Logger must be an instance of CylinderLogger")
    else:
        DEFAULT_LOGGER = logger
