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

    config_class : pydantic.BaseModel = None
    name : str = "base"
    default_config = None

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        pass

_RDB_PREPROCESS_REGISTRY = {}

def rdb_preprocess(preprocess_class):
    global _RDB_PREPROCESS_REGISTRY
    _RDB_PREPROCESS_REGISTRY[preprocess_class.name] = preprocess_class
    return preprocess_class

def get_rdb_preprocess_class(name : str):
    global _RDB_PREPROCESS_REGISTRY
    preprocess_class = _RDB_PREPROCESS_REGISTRY.get(name, None)
    if preprocess_class is None:
        raise ValueError(f"Cannot find the preprocess class of name {name}.")
    return preprocess_class

def get_rdb_preprocess_choice():
    names = _RDB_PREPROCESS_REGISTRY.keys()
    return Enum("RDBPreprocessChoice", {name.upper() : name for name in names})
