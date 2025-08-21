"""
Transform utilities and convenience functions.

This module provides utility functions to make it easier to use transforms
with the current RDB dataset structure.
"""

from typing import Union, List
from .base import RDBTransform, TableTransform, ColumnTransform, RDBTransformWrapper


def apply_transform(rdb, transform: Union[RDBTransform, TableTransform, ColumnTransform]):
    """
    Apply a transform to an RDB dataset.
    
    Args:
        rdb: DBBRDBDataset to transform
        transform: Transform to apply (RDBTransform, TableTransform, or ColumnTransform)
        
    Returns:
        Transformed RDB dataset
    """
    if isinstance(transform, RDBTransform):
        return transform(rdb)
    elif isinstance(transform, (TableTransform, ColumnTransform)):
        wrapper = RDBTransformWrapper(transform)
        return wrapper(rdb)
    else:
        raise ValueError(f"Unknown transform type: {type(transform)}")


def apply_transform_pipeline(rdb, transforms: List[Union[RDBTransform, TableTransform, ColumnTransform]]):
    """
    Apply a sequence of transforms to an RDB dataset.
    
    Args:
        rdb: DBBRDBDataset to transform
        transforms: List of transforms to apply in sequence
        
    Returns:
        Transformed RDB dataset
    """
    result = rdb
    for transform in transforms:
        result = apply_transform(result, transform)
    return result
