# FastDFS - Deep Feature Synthesis for Tabular Data
"""
FastDFS - Fast Deep Feature Synthesis

A standalone package for deep feature synthesis extracted from tab2graph.
"""

__version__ = "0.1.0"

# Import enhanced API for easy access
from .api import (
    run_dfs,
    run_transform, 
    run_full_pipeline,
    load_dataset,
    save_dataset
)

__all__ = [
    "run_dfs",
    "run_transform",
    "run_full_pipeline", 
    "load_dataset",
    "save_dataset"
]

__version__ = "0.1.0"

# Main functions
from .dataset import load_rdb_data, DBBRDBDataset, DBBRDBTask

# Core classes
from .preprocess.dfs import DFSPreprocess, DFSConfig
from .preprocess.transform_preprocess import RDBTransformPreprocess
from .preprocess import get_rdb_preprocess_class, get_rdb_preprocess_choice

# Utilities
from .utils import DeviceInfo

__all__ = [
    "load_rdb_data",
    "DBBRDBDataset", 
    "DBBRDBTask",
    "DFSPreprocess",
    "DFSConfig", 
    "RDBTransformPreprocess",
    "get_rdb_preprocess_class",
    "get_rdb_preprocess_choice",
    "DeviceInfo",
]
