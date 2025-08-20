"""
Base classes for RDB transformations.

This module provides the core transform interface classes that define the 
new simplified functional approach to data transformations in FastDFS.
"""

import abc
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import pandas as pd

from ..dataset.meta import DBBColumnSchema
from ..dataset.rdb_simplified import RDBTableSchema

if TYPE_CHECKING:
    from ..dataset.rdb_simplified import RDBDataset


class RDBTransform(abc.ABC):
    """Base class for RDB transformations - simplified composable operations."""
    
    @abc.abstractmethod
    def __call__(self, rdb: 'RDBDataset') -> 'RDBDataset':
        """
        Apply transformation to RDB and return new RDB.
        
        Simplified interface - no separate fit/transform phases.
        Each transform is a pure function: RDB -> RDB
        
        Args:
            rdb: RDBDataset to transform
            
        Returns:
            New RDBDataset with transformation applied
        """
        pass


class TableTransform(abc.ABC):
    """Transform that operates on individual tables within an RDB."""
    
    @abc.abstractmethod
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> Tuple[pd.DataFrame, RDBTableSchema]:
        """
        Apply transformation to a single table.
        
        Args:
            table: DataFrame to transform
            table_metadata: Metadata for the table
            
        Returns:
            Tuple of (transformed_dataframe, updated_table_metadata)
            
        Expected Behavior:
        - For unchanged tables: return (original_table, original_metadata)
        - For modified tables: return (modified_table, updated_metadata with changed column schemas)
        - For tables with new columns: return (table_with_new_cols, metadata with additional column schemas)
        - For tables with removed columns: return (table_without_cols, metadata with removed column schemas)
        """
        pass


class ColumnTransform(abc.ABC):
    """Transform that operates on specific columns matching criteria."""
    
    @abc.abstractmethod
    def applies_to(self, column_metadata: DBBColumnSchema) -> bool:
        """Check if this transform should be applied to a column."""
        pass
    
    @abc.abstractmethod
    def __call__(self, column: pd.Series, column_metadata: DBBColumnSchema) -> Tuple[pd.DataFrame, List[DBBColumnSchema]]:
        """
        Transform a column, potentially outputting multiple new columns.
        
        Args:
            column: Series to transform
            column_metadata: Metadata for the column
            
        Returns:
            Tuple of (dataframe_with_new_columns, list_of_new_column_schemas)
            
        Expected Behavior:
        - For unchanged columns: return (df_with_original_column, [original_column_schema])
        - For modified columns: return (df_with_modified_column, [updated_column_schema])
        - For columns generating new features: return (df_with_multiple_columns, [schema_for_each_new_column])
        - For removed columns: should consider using TableTransform instead
        """
        pass


class RDBTransformPipeline(RDBTransform):
    """Pipeline of RDB transformations."""
    
    def __init__(self, transforms: List[RDBTransform]):
        """
        Initialize pipeline with list of transforms.
        
        Args:
            transforms: List of RDBTransform objects to apply in sequence
        """
        self.transforms = transforms
    
    def __call__(self, rdb: 'RDBDataset') -> 'RDBDataset':
        """Apply all transforms in sequence."""
        result = rdb
        for transform in self.transforms:
            result = transform(result)
        return result


class RDBTransformWrapper(RDBTransform):
    """Wrapper to apply TableTransform or ColumnTransform to entire RDB."""
    
    def __init__(self, inner_transform: Union[TableTransform, ColumnTransform]):
        """
        Initialize wrapper with a table or column transform.
        
        Args:
            inner_transform: TableTransform or ColumnTransform to wrap
        """
        self.inner_transform = inner_transform
    
    def __call__(self, rdb: 'RDBDataset') -> 'RDBDataset':
        """Apply inner transform to all applicable tables/columns in RDB."""
        new_tables = {}
        updated_metadata = {}
        
        for table_name in rdb.tables.keys():
            table_df = rdb.get_table(table_name)
            table_metadata = rdb.get_table_metadata(table_name)
            
            if isinstance(self.inner_transform, TableTransform):
                # Apply to entire table
                new_table, new_table_metadata = self.inner_transform(table_df, table_metadata)
                new_tables[table_name] = new_table
                updated_metadata[table_name] = new_table_metadata
                
            elif isinstance(self.inner_transform, ColumnTransform):
                # Apply to applicable columns
                new_table = table_df.copy()
                new_column_schemas = []
                
                for col_schema in table_metadata.columns:
                    if self.inner_transform.applies_to(col_schema):
                        col_name = col_schema.name
                        if col_name in table_df.columns:
                            new_cols_df, new_col_schemas = self.inner_transform(table_df[col_name], col_schema)
                            
                            # Add/replace columns from transform result
                            for new_col_name, new_col_data in new_cols_df.items():
                                new_table[new_col_name] = new_col_data
                            
                            # Update column schemas
                            new_column_schemas.extend(new_col_schemas)
                            
                            # Remove original column if transform doesn't retain it
                            if hasattr(self.inner_transform, 'replaces_original') and self.inner_transform.replaces_original:
                                if col_name in new_table.columns:
                                    new_table = new_table.drop(columns=[col_name])
                        else:
                            # Keep original schema if column doesn't exist in data
                            new_column_schemas.append(col_schema)
                    else:
                        # Keep original column and schema if transform doesn't apply
                        new_column_schemas.append(col_schema)
                
                new_tables[table_name] = new_table
                
                # Create updated table metadata
                updated_metadata[table_name] = RDBTableSchema(
                    name=table_metadata.name,
                    source=table_metadata.source,
                    format=table_metadata.format,
                    columns=new_column_schemas,
                    time_column=table_metadata.time_column
                )
        
        return rdb.create_new_with_tables_and_metadata(new_tables, updated_metadata)
