"""
Centralized logging configuration for FastDFS using loguru.

This module provides a simple way to configure loguru for the entire project.
It should be imported once at the application entry point.
"""

import sys
from loguru import logger
from typing import Optional


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    serialize: bool = False
) -> None:
    """
    Configure loguru logger for FastDFS.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses default.
        serialize: Whether to output logs in JSON format
    """
    # Remove default handler
    logger.remove()
    
    # Default format if none provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    
    # Add handler with configuration
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        serialize=serialize,
        backtrace=True,
        diagnose=True
    )


def configure_file_logging(
    file_path: str,
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "10 days",
    compression: str = "zip"
) -> None:
    """
    Add file logging configuration.
    
    Args:
        file_path: Path to log file
        level: Log level for file output
        rotation: When to rotate log files
        retention: How long to keep old logs
        compression: Compression format for old logs
    """
    logger.add(
        file_path,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True
    )


# For backward compatibility, configure with INFO level by default
# This can be called during module import or at application startup
configure_logging()
