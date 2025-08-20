"""
Featuretools-based DFS engine implementation for the new interface.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import featuretools as ft
import tqdm
from loguru import logger

from .base_engine import DFSEngine, DFSConfig, dfs_engine
from ..dataset.rdb_simplified import RDBDataset

__all__ = ['FeaturetoolsEngine']


@dfs_engine
class FeaturetoolsEngine(DFSEngine):
    """Featuretools-based DFS engine implementation."""
    
    name = "featuretools"
    
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using featuretools (reuse existing computation logic)."""
        
        # Rebuild EntitySet for computation (could be optimized to reuse from prepare phase)
        entity_set = self._build_entity_set_from_rdb(rdb)
        target_entity_name = "__target__"
        target_index = "__target_index__"  # Target index is already handled by base class
        
        # Handle cutoff time processing
        target_df_copy = target_dataframe.copy(deep=False)
        cutoff_times = None
        
        if cutoff_time_column and config.use_cutoff_time:
            # Create separate dataframe for cutoff times with index and timestamps
            cutoff_times = target_df_copy[[target_index, cutoff_time_column]].copy(deep=False)
            cutoff_times.columns = ["instance_id", "time"]
            
            # Temporarily drop the time column from target dataframe
            target_df_copy = target_df_copy.drop(columns=[cutoff_time_column])
        
        # Add target dataframe to EntitySet (without time column if cutoff is used)
        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_df_copy,
            index=target_index,
            time_index=None  # No time index on target since we handle cutoff separately
        )
        
        # Add relationships from target to RDB entities
        self._add_target_relationships(entity_set, target_entity_name, key_mappings)
        
        # Compute feature matrix using featuretools with strict cutoff behavior
        with tqdm.tqdm(total=100) as pbar:
            
            def _cb(update, progress_percent, time_elapsed):
                pbar.update(int(update))
            
            feature_matrix = ft.calculate_feature_matrix(
                features=features,
                entityset=entity_set,
                cutoff_time=cutoff_times,
                include_cutoff_time=False,  # Use strict < for cutoff time
                chunk_size=config.chunk_size,
                n_jobs=config.n_jobs,
                progress_callback=_cb
            )
        
        # Reset index to get target_index as a column for sorting
        if target_index in feature_matrix.index.names or target_index == feature_matrix.index.name:
            feature_matrix = feature_matrix.reset_index()
        elif target_index not in feature_matrix.columns:
            # Add the target index column if somehow missing
            feature_matrix[target_index] = range(len(feature_matrix))
        
        # Sort by target index to ensure consistent row order
        feature_matrix = feature_matrix.sort_values(by=target_index).reset_index(drop=True)
        
        # Since both dataframes are sorted by target_index, we can concatenate along column axis
        # First drop target_index from feature_matrix to avoid duplication
        feature_columns = feature_matrix.drop(columns=[target_index])
        
        # Concatenate along column axis (axis=1)
        result = pd.concat([target_dataframe, feature_columns], axis=1)
        
        return result
