"""
Type canonicalization transform.

This module implements CanonicalizeTypes, a TableTransform that enforces
data types in RDB tables according to their metadata schema.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from loguru import logger

from ..dataset.meta import RDBTableSchema, RDBColumnDType
from .base import TableTransform


class CanonicalizeTypes(TableTransform):
    """
    Enforce data types in DataFrame columns according to RDB metadata.
    
    This transform iterates through columns defined in the table schema and
    casts the DataFrame columns to the appropriate pandas/numpy types.
    """
    
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]:
        """
        Apply type canonicalization to a table.
        
        Args:
            table: DataFrame to transform
            table_metadata: Metadata for the table
            
        Returns:
            Tuple of (transformed_dataframe, original_metadata)
            
        Raises:
            ValueError: If a column defined in metadata is missing from the table.
            
        Note:
            Columns present in the table but NOT in the metadata will be dropped.
        """
        new_table = pd.DataFrame()
        
        for col_schema in table_metadata.columns:
            col_name = col_schema.name
            
            if col_name not in table.columns:
                raise ValueError(
                    f"Column '{col_name}' defined in metadata for table '{table_metadata.name}' "
                    f"is missing from the actual data."
                )
                
            dtype = col_schema.dtype
            
            try:
                if dtype == RDBColumnDType.float_t:
                    # Cast to float32, handling errors by coercing to NaN
                    new_table[col_name] = pd.to_numeric(table[col_name], errors='coerce').astype(np.float32)
                    
                elif dtype == RDBColumnDType.datetime_t:
                    # Cast to datetime64[ns]
                    new_table[col_name] = pd.to_datetime(table[col_name], errors='coerce')
                    
                elif dtype == RDBColumnDType.timestamp_t:
                    # Cast to int64 (timestamp)
                    # First convert to numeric (handles strings), then fill NaNs, then cast to int
                    # Note: int64 doesn't support NaN, so we might need Int64 (nullable) or fillna
                    # Here we use Int64 to allow nulls
                    new_table[col_name] = pd.to_numeric(table[col_name], errors='coerce').astype('Int64')
                    
                elif dtype == RDBColumnDType.text_t:
                    # Cast to string/object
                    new_table[col_name] = table[col_name].astype(str)
                    
                elif dtype == RDBColumnDType.category_t:
                    # Cast to category
                    new_table[col_name] = table[col_name].astype('category')
                    
                elif dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]:
                    # Keys are typically strings or integers, but 'object' is safest for mixed types
                    # or to preserve exact formatting of IDs
                    new_table[col_name] = table[col_name].astype(str)
                    
            except Exception as e:
                logger.warning(f"Failed to cast column '{col_name}' in table '{table_metadata.name}' to {dtype}: {e}")
        
        # Metadata remains unchanged as we are only enforcing the types defined in it
        return new_table, table_metadata
