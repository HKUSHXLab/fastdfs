# FastDFS - Deep Feature Synthesis for Tabular Data
"""
FastDFS - Fast Deep Feature Synthesis

A standalone package for deep feature synthesis using a table-centric approach.
"""

__version__ = "0.2.0"

from loguru import logger
logger.disable("fastdfs")

# Main Table-Centric API
from .api import (
    load_rdb,
    compute_dfs_features,
    DFSPipeline
)

# Core components
from .dfs import DFSConfig
from .dataset.rdb import RDB, RDBDataset
from .transform import (
    RDBTransform,
    RDBTransformPipeline,
    RDBTransformWrapper,
    FeaturizeDatetime, 
    FilterColumn, 
    HandleDummyTable
)

__all__ = [
    # Main API
    "load_rdb",
    "compute_dfs_features", 
    "DFSPipeline",
    
    # Core components
    "DFSConfig",
    "RDB",
    "RDBDataset",
    "RDBTransform",
    "RDBTransformPipeline",
    "RDBTransformWrapper",
    "FeaturizeDatetime",
    "FilterColumn", 
    "HandleDummyTable"
]
