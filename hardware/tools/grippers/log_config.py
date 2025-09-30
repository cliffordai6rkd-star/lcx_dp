"""Unified logging configuration for HIROLRobotPlatform.

This module provides unified logging configuration to prevent duplicate log outputs
from both Python logging and glog systems.
"""

import logging
import sys
from typing import Optional


def setup_unified_logging(
    level: str = "INFO",
    enable_glog: bool = True,
    enable_python_logging: bool = True,
    log_format: Optional[str] = None,
    disable_duplicate_handlers: bool = True
) -> None:
    """Configure unified logging system for HIROLRobotPlatform.
    
    This function sets up a unified logging configuration that prevents
    duplicate log outputs between Python logging and glog systems.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        enable_glog: Whether to enable glog output.
        enable_python_logging: Whether to enable Python logging output.
        log_format: Custom log format string. If None, uses default format.
        disable_duplicate_handlers: Whether to remove duplicate handlers.
        
    Raises:
        ValueError: If level parameter is invalid.
        RuntimeError: If logging system initialization fails.
    """
    # Validate level parameter
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {valid_levels}")
    
    try:
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper())
        
        # Set default log format if not provided
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Clear existing handlers to prevent duplicates
        if disable_duplicate_handlers:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        # Configure Python logging if enabled
        if enable_python_logging:
            logging.basicConfig(
                level=numeric_level,
                format=log_format,
                stream=sys.stdout,
                force=True  # Force reconfiguration
            )
        
        # Set specific logger levels for Pika SDK modules
        pika_loggers = [
            "pika.gripper",
            "pika.serial_comm", 
            "pika.sense",
            "pika.camera.fisheye",
            "pika.camera.realsense",
            "pika.tracker.vive_tracker"
        ]
        
        for logger_name in pika_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)
            # Prevent propagation to avoid duplicate messages
            if disable_duplicate_handlers:
                logger.propagate = True
        
    except Exception as e:
        raise RuntimeError(f"Failed to setup unified logging: {e}") from e


def set_pika_logging_level(level: str) -> None:
    """Set logging level for Pika SDK related modules.
    
    Args:
        level: Target logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        
    Raises:
        ValueError: If level parameter is invalid.
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {valid_levels}")
    
    numeric_level = getattr(logging, level.upper())
    
    pika_loggers = [
        "pika.gripper",
        "pika.serial_comm",
        "pika.sense", 
        "pika.camera.fisheye",
        "pika.camera.realsense",
        "pika.tracker.vive_tracker"
    ]
    
    for logger_name in pika_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)


def suppress_pika_info_logs() -> None:
    """Suppress INFO level logs from Pika SDK modules to reduce output noise."""
    set_pika_logging_level("WARNING")


def enable_debug_logging() -> None:
    """Enable DEBUG level logging for troubleshooting."""
    setup_unified_logging(level="DEBUG")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration.
    
    Args:
        name: Logger name.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)