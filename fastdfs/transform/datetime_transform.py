"""
Datetime feature extraction transform.

This module implements FeaturizeDatetime, a ColumnTransform that extracts 
datetime features based on the original implementation in 
fastdfs/utils/datetime_utils.py.
"""

from typing import List, Tuple
import pandas as pd
from ..dataset.meta import DBBColumnSchema, DBBColumnDType, RDBTableSchema
from ..utils.datetime_utils import featurize_datetime_column
from .base import ColumnTransform


class FeaturizeDatetime(ColumnTransform):
    """Extract datetime features from datetime columns."""
    
    def __init__(self, features: List[str] = None):
        """
        Initialize datetime featurizer.
        
        Args:
            features: List of features to extract. 
                     Options: ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek']
                     Default extracts: ['year', 'month', 'day', 'hour']
        """
        self.features = features or ['year', 'month', 'day', 'hour']
    
    def applies_to(self, column_metadata: DBBColumnSchema) -> bool:
        """Check if this transform should be applied to datetime columns."""
        return column_metadata.dtype == DBBColumnDType.datetime_t
    
    def __call__(self, column: pd.Series, column_metadata: DBBColumnSchema) -> Tuple[pd.DataFrame, List[DBBColumnSchema]]:
        """Extract datetime features from a datetime column."""
        
        # Check if this is a datetime column
        if column_metadata.dtype != DBBColumnDType.datetime_t:
            # Return original column as DataFrame
            return pd.DataFrame({column_metadata.name: column}), [column_metadata]
        
        # Use existing featurization utility
        feature_df = featurize_datetime_column(column, self.features)
        
        # Create new column schemas for each feature
        new_column_schemas = []
        base_name = column_metadata.name
        
        for feature_name in feature_df.columns:
            if feature_name != base_name:  # Skip original column if included
                new_col_schema = DBBColumnSchema(
                    name=feature_name,
                    dtype=DBBColumnDType.float_t  # Datetime features are typically numeric
                )
                new_column_schemas.append(new_col_schema)
        
        # Include original column if not replaced
        if base_name in feature_df.columns:
            new_column_schemas.insert(0, column_metadata)
        
        return feature_df, new_column_schemas
