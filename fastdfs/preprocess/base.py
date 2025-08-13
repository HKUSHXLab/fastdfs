from enum import Enum
from typing import Tuple, Dict, Optional, List
import abc
from pathlib import Path
import pydantic

from ..dataset import (
    DBBColumnDType,
    DBBRDBDataset,
)

from ..utils.device import DeviceInfo

class RDBDatasetPreprocess:
    """
    Abstract base class for RDB dataset preprocessing pipelines.
    
    This class provides the foundation for implementing data preprocessing workflows
    that operate on entire RDB datasets. It handles the I/O operations and coordinates
    the execution of preprocessing algorithms.
    
    The class follows a configuration-driven approach where preprocessing behavior
    is controlled through Pydantic configuration models specified via config_class.
    
    Attributes:
        config_class: Pydantic model class for configuration validation
        name: String identifier for this preprocessor type
        default_config: Default configuration instance for this preprocessor
    """

    config_class : pydantic.BaseModel = None
    name : str = "base"
    default_config = None

    def __init__(self, config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration object that should be an instance of config_class
        """
        self.config = config

    @abc.abstractmethod
    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        """
        Execute the preprocessing pipeline on the given dataset.
        
        This method performs the core preprocessing logic, transforming the input
        dataset and writing the results to the specified output path.
        
        Args:
            dataset: Input RDB dataset to preprocess
            output_path: Directory path where the preprocessed dataset will be saved
            device: Device information for hardware-specific optimizations
        """
        pass

_RDB_PREPROCESS_REGISTRY = {}

def rdb_preprocess(preprocess_class):
    """
    Decorator to register an RDB preprocessor class.
    
    This decorator adds the preprocessor class to the global registry, making it
    available for lookup by name through get_rdb_preprocess_class().
    
    Args:
        preprocess_class: Class that inherits from RDBDatasetPreprocess
        
    Returns:
        The same class, now registered in the global registry
        
    Example:
        @rdb_preprocess
        class MyPreprocess(RDBDatasetPreprocess):
            name = "my_preprocess"
    """
    global _RDB_PREPROCESS_REGISTRY
    _RDB_PREPROCESS_REGISTRY[preprocess_class.name] = preprocess_class
    return preprocess_class

def get_rdb_preprocess_class(name : str):
    """
    Retrieve a registered RDB preprocessor class by name.
    
    Args:
        name: Name of the preprocessor class to retrieve
        
    Returns:
        The preprocessor class
        
    Raises:
        ValueError: If no preprocessor with the given name is registered
    """
    global _RDB_PREPROCESS_REGISTRY
    preprocess_class = _RDB_PREPROCESS_REGISTRY.get(name, None)
    if preprocess_class is None:
        raise ValueError(f"Cannot find the preprocess class of name {name}.")
    return preprocess_class

def get_rdb_preprocess_choice():
    """
    Get an Enum containing all registered preprocessor names.
    
    Returns:
        Enum with all available preprocessor names as choices
    """
    names = _RDB_PREPROCESS_REGISTRY.keys()
    return Enum("RDBPreprocessChoice", {name.upper() : name for name in names})
