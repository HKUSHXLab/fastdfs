"""
Minimal API for the table-centric DFS interface.

This module provides the core functions for computing DFS features using
the RDB interface.
"""

from typing import Dict, Optional, Any, List, Tuple
import pandas as pd
from pathlib import Path
from loguru import logger

from .dfs import DFSConfig, get_dfs_engine
from .dataset.rdb import RDB
from .dataset.meta import RDBMeta, RDBTableSchema, RDBTableDataFormat, RDBColumnDType
from .transform.infer_schema import InferSchemaTransform

__all__ = ['load_rdb', 'create_rdb', 'compute_dfs_features', 'DFSPipeline']


def load_rdb(path: str) -> RDB:
    """
    Load a relational database.
    
    Args:
        path: Path to the RDB directory
        
    Returns:
        RDB instance
    """
    return RDB(Path(path))


def create_rdb(
    tables: Dict[str, pd.DataFrame],
    name: str = "myrdb",
    primary_keys: Optional[Dict[str, str]] = None,
    foreign_keys: Optional[List[Tuple[str, str, str, str]]] = None,
    time_columns: Optional[Dict[str, str]] = None,
    type_hints: Optional[Dict[str, Dict[str, str]]] = None
) -> RDB:
    """
    Create an RDB from a dictionary of pandas DataFrames.
    
    This function automatically infers the schema (column types) from the dataframes,
    using the provided metadata (keys, time columns) as hints.
    
    Args:
        tables: Dictionary mapping table names to pandas DataFrames.
        name: Name of the RDB.
        primary_keys: Dictionary mapping table names to their primary key column name.
        foreign_keys: List of relationships as (child_table, child_col, parent_table, parent_col).
        time_columns: Dictionary mapping table names to their time column name.
        type_hints: Dictionary mapping table names to a dictionary of {column_name: dtype_str}.
                    Useful for overriding inferred types.
                    
    Returns:
        RDB: An initialized RDB object with inferred schema.
    """
    # Create initial empty schemas
    table_schemas = []
    for table_name in tables.keys():
        # We create a minimal schema with just the name and format
        # The columns will be populated by the transform
        schema = RDBTableSchema(
            name=table_name,
            source=f"{table_name}.parquet", # Placeholder
            format=RDBTableDataFormat.PARQUET,
            columns=[] # Empty initially
        )
        table_schemas.append(schema)
        
    metadata = RDBMeta(
        name=name,
        tables=table_schemas
    )
    
    rdb = RDB(metadata=metadata, tables=tables)
    
    # Apply inference transform
    transform = InferSchemaTransform(
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        time_columns=time_columns,
        type_hints=type_hints
    )
    
    rdb = transform(rdb)
    
    return rdb


def compute_dfs_features(
    rdb: RDB,
    target_dataframe: pd.DataFrame,
    key_mappings: Dict[str, str],
    cutoff_time_column: Optional[str] = None,
    config: Optional[DFSConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compute DFS features for a target dataframe using RDB context.
    
    Args:
        rdb: The relational database providing context for feature generation
        target_dataframe: DataFrame to augment with features
        key_mappings: Map from target_dataframe columns to RDB primary keys
                     e.g., {"user_id": "user.user_id", "item_id": "item.item_id"}
        cutoff_time_column: Column name in target_dataframe for temporal cutoff
        config: DFS configuration (uses defaults if not provided)
        config_overrides: Dictionary of config parameters to override
        
    Returns:
        DataFrame with original target_dataframe data plus generated features
        
    Example:
        >>> rdb = load_rdb("ecommerce_rdb/")
        >>> target_df = pd.DataFrame({
        ...     "user_id": [1, 2, 3],
        ...     "item_id": [100, 200, 300],
        ...     "interaction_time": ["2024-01-01", "2024-01-02", "2024-01-03"]
        ... })
        >>> features_df = compute_dfs_features(
        ...     rdb=rdb,
        ...     target_dataframe=target_df,
        ...     key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
        ...     cutoff_time_column="interaction_time",
        ...     config_overrides={"max_depth": 2, "engine": "dfs2sql"}
        ... )
    """
    # Use default config if not provided
    if config is None:
        config = DFSConfig()
    
    # Validate key types
    for target_col, rdb_key in key_mappings.items():
        if target_col not in target_dataframe.columns:
            raise ValueError(f"Key column '{target_col}' not found in target dataframe.")
            
        table_name, col_name = rdb_key.split('.')
        try:
            table_meta = rdb.get_table_metadata(table_name)
        except ValueError:
            raise ValueError(f"Table '{table_name}' not found in RDB.")
            
        if col_name not in table_meta.column_dict:
            raise ValueError(f"Column '{col_name}' not found in table '{table_name}'.")
            
        col_meta = table_meta.column_dict[col_name]

        if col_meta.dtype != RDBColumnDType.primary_key:
            raise ValueError(f"RDB column '{rdb_key}' is not a primary key. Key mappings must point to primary keys.")

        if col_meta.dtype in (RDBColumnDType.primary_key, RDBColumnDType.foreign_key):
            # Convert target column to string to match RDB key columns (which should always be string)
            target_col_dtype = target_dataframe[target_col].dtype
            
            # Convert to string if not already string (categorical is converted for consistency)
            if not pd.api.types.is_string_dtype(target_col_dtype):
                try:
                    target_dataframe[target_col] = target_dataframe[target_col].astype(str)
                    
                    # Log the conversion for transparency
                    logger.debug(
                        f"Converted target column '{target_col}' from {target_col_dtype} to string "
                        f"to match RDB key '{rdb_key}'"
                    )
                except (ValueError, TypeError) as e:
                    # If conversion fails, raise a helpful error
                    raise TypeError(
                        f"Failed to convert column '{target_col}' from {target_col_dtype} to string "
                        f"to match RDB key '{rdb_key}'. Original error: {e}. "
                        f"Please ensure the values are compatible or manually convert: "
                        f"target_df['{target_col}'] = target_df['{target_col}'].astype(str)"
                    )

    # Get the appropriate engine
    engine = get_dfs_engine(config.engine, config)
    
    # Compute features
    return engine.compute_features(
        rdb=rdb,
        target_dataframe=target_dataframe,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff_time_column,
        config_overrides=config_overrides
    )


class DFSPipeline:
    """
    Pipeline for combining RDB transforms with DFS feature computation.
    
    This class allows you to compose preprocessing transforms with feature
    generation in a single pipeline.
    """
    
    def __init__(
        self,
        transform_pipeline = None,
        dfs_config: Optional[DFSConfig] = None
    ):
        """
        Initialize the DFS pipeline.
        
        Args:
            transform_pipeline: RDB transform pipeline (optional)
            dfs_config: DFS configuration (uses defaults if not provided)
        """
        self.transform_pipeline = transform_pipeline
        self.dfs_config = dfs_config or DFSConfig()
    
    def compute_features(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply transforms to RDB and compute DFS features.
        
        Args:
            rdb: Input RDB dataset
            target_dataframe: DataFrame to augment with features
            key_mappings: Map from target_dataframe columns to RDB primary keys
            cutoff_time_column: Column name for temporal cutoff
            config_overrides: Dictionary of config parameters to override
            
        Returns:
            DataFrame with features computed from transformed RDB
        """
        # Apply transforms if provided
        if self.transform_pipeline is not None:
            transformed_rdb = self.transform_pipeline(rdb)
        else:
            transformed_rdb = rdb
        
        # Compute features using transformed RDB
        return compute_dfs_features(
            rdb=transformed_rdb,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=cutoff_time_column,
            config=self.dfs_config,
            config_overrides=config_overrides
        )