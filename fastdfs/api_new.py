"""
High-level API for the new table-centric DFS interface.

This module provides convenient functions for computing DFS features using
the new simplified RDB dataset interface.
"""

from typing import Dict, Optional, Any
import pandas as pd
from pathlib import Path

from .dfs import DFSConfig, get_dfs_engine
from .dataset.rdb_simplified import RDBDataset

__all__ = ['load_rdb', 'compute_dfs_features', 'DFSPipeline']


def load_rdb(path: str) -> RDBDataset:
    """
    Load a relational database dataset.
    
    Args:
        path: Path to the RDB dataset directory
        
    Returns:
        RDBDataset instance
    """
    return RDBDataset(Path(path))


def compute_dfs_features(
    rdb: RDBDataset,
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
        rdb: RDBDataset,
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
