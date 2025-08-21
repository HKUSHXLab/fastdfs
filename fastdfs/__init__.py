# FastDFS - Deep Feature Synthesis for Tabular Data
"""
FastDFS - Fast Deep Feature Synthesis

A standalone package for deep feature synthesis using a table-centric approach.
"""

__version__ = "0.1.0"

# Main Table-Centric API
from .api import (
    load_rdb,
    compute_dfs_features,
    DFSPipeline
)

# Core components
from .dfs import DFSConfig
from .dataset.rdb_simplified import RDBDataset
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
    "RDBDataset",
    "RDBTransform",
    "RDBTransformPipeline",
    "RDBTransformWrapper",
    "FeaturizeDatetime",
    "FilterColumn", 
    "HandleDummyTable"
]
