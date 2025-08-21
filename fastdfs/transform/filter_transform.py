"""
Column filtering transform.

This module implements FilterColumn, a TableTransform that removes redundant
and unwanted columns based on the original implementation in 
fastdfs/preprocess/transform/filter_column.py.
"""

from typing import List, Optional, Tuple
import pandas as pd
from ..dataset.meta import RDBTableSchema, RDBColumnDType
from .base import TableTransform


class FilterColumn(TableTransform):
    """Filter redundant and unwanted columns from tables."""
    
    def __init__(self, drop_dtypes: Optional[List[str]] = None, drop_redundant: bool = True):
        """
        Initialize column filter.
        
        Args:
            drop_dtypes: List of data types to remove (e.g., ['text'])
            drop_redundant: Whether to remove columns with only identical values
        """
        self.drop_dtypes = drop_dtypes or []
        self.drop_redundant = drop_redundant
    
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]:
        """Remove redundant or unwanted columns from table."""
        columns_to_drop = []
        
        for col_schema in table_metadata.columns:
            col_name = col_schema.name
            
            # Never drop primary or foreign keys
            if col_schema.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]:
                continue
                
            # Skip vector embeddings (multi-dimensional data)
            if col_name in table.columns:
                col_data = table[col_name]
                # Check if any values are array-like (exclude strings)
                try:
                    if col_data.apply(lambda x: hasattr(x, '__len__') and not isinstance(x, str) and len(x) > 1).any():
                        continue
                except:
                    pass  # If check fails, continue with normal processing
                
            # Drop redundant columns (single unique value)
            if self.drop_redundant and col_schema.dtype != RDBColumnDType.text_t:
                if col_name in table.columns:
                    unique_values = table[col_name].nunique(dropna=False)
                    if unique_values <= 1:
                        columns_to_drop.append(col_name)
                        continue
            
            # Drop columns by data type
            if col_schema.dtype in self.drop_dtypes:
                columns_to_drop.append(col_name)
        
        # Create new table and metadata
        new_table = table.drop(columns=columns_to_drop, errors='ignore')
        new_column_schemas = [
            col_schema for col_schema in table_metadata.columns 
            if col_schema.name not in columns_to_drop
        ]
        
        new_metadata = RDBTableSchema(
            name=table_metadata.name,
            source=table_metadata.source,
            format=table_metadata.format,
            columns=new_column_schemas,
            time_column=table_metadata.time_column
        )
        
        return new_table, new_metadata
