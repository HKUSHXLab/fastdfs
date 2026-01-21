"""
Datetime feature extraction transform.

This module implements FeaturizeDatetime, a ColumnTransform that extracts 
datetime features based on the original implementation in 
fastdfs/utils/datetime_utils.py.
"""

from typing import List, Tuple
import pandas as pd
from ..dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableSchema
from ..utils.datetime_utils import featurize_datetime_column
from .base import ColumnTransform


class FeaturizeDatetime(ColumnTransform):
    """Extract datetime features from datetime columns."""
    
    def __init__(self, features: List[str] = None, retain_original: bool = True):
        """
        Initialize datetime featurizer.

        Args:
            features: List of features to extract.
                     Options: ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'timestamp']
                     - 'timestamp' extracts epoch time in nanoseconds (numeric representation)
                     Default extracts: ['year', 'month', 'day', 'hour']
            retain_original: Whether to keep the original datetime column. Default is True.
        """
        self.features = features or ['year', 'month', 'day', 'hour']
        self.retain_original = retain_original
    
    def applies_to(self, column_metadata: RDBColumnSchema) -> bool:
        """Check if this transform should be applied to datetime columns."""
        return column_metadata.dtype == RDBColumnDType.datetime_t
    
    def __call__(self, column: pd.Series, column_metadata: RDBColumnSchema) -> Tuple[pd.DataFrame, List[RDBColumnSchema]]:
        """Extract datetime features from a datetime column."""
        
        # Check if this is a datetime column
        if column_metadata.dtype != RDBColumnDType.datetime_t:
            # Return original column as DataFrame
            return pd.DataFrame({column_metadata.name: column}), [column_metadata]
        
        # Use existing featurization utility
        feature_df = featurize_datetime_column(column, self.features)
        
        # Create new column schemas for each feature
        new_column_schemas = []
        base_name = column_metadata.name
        
        for feature_name in feature_df.columns:
            if feature_name != base_name:  # Skip original column if included
                new_col_schema = RDBColumnSchema(
                    name=feature_name,
                    dtype=RDBColumnDType.float_t  # Datetime features are typically numeric
                )
                new_column_schemas.append(new_col_schema)
        
        # Handle original column retention
        if self.retain_original:
            if base_name not in feature_df.columns:
                # Add it back if featurize_datetime_column didn't return it
                feature_df[base_name] = column
            new_column_schemas.insert(0, column_metadata)
        else:
            if base_name in feature_df.columns:
                feature_df = feature_df.drop(columns=[base_name])
        
        return feature_df, new_column_schemas
