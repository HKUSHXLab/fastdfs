"""
Test to verify that the logging refactoring with loguru works correctly.
"""

import pytest
from loguru import logger
import io
import sys


def test_loguru_import():
    """Test that loguru can be imported and used."""
    # This should not raise any exceptions
    from loguru import logger
    logger.info("Test message")


def test_fastdfs_api_import():
    """Test that FastDFS API can be imported with loguru."""
    # This should not raise any exceptions
    import fastdfs.api
    import fastdfs.preprocess.dfs.dfs_preprocess
    import fastdfs.preprocess.transform_preprocess


def test_logger_functionality():
    """Test that logger works correctly."""
    # Capture log output
    stream = io.StringIO()
    
    # Remove default handler and add our test handler
    logger.remove()
    logger.add(stream, format="{level} | {message}")
    
    # Test different log levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Get the output
    output = stream.getvalue()
    
    # Verify all messages are present
    assert "DEBUG | Debug message" in output
    assert "INFO | Info message" in output
    assert "WARNING | Warning message" in output
    assert "ERROR | Error message" in output


def test_logging_config_module():
    """Test that the logging configuration module works."""
    from fastdfs.utils.logging_config import configure_logging, configure_file_logging
    
    # These should not raise exceptions
    configure_logging(level="DEBUG")
    
    
if __name__ == "__main__":
    pytest.main([__file__])
