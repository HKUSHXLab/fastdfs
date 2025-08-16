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
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        
        # Add target dataframe to EntitySet
        target_df_copy = target_dataframe.copy()
        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_df_copy,
            index=target_index,
            time_index=cutoff_time_column
        )
        
        # Add relationships from target to RDB entities
        self._add_target_relationships(entity_set, target_entity_name, key_mappings)
        
        # Prepare cutoff times if needed
        cutoff_times = None
        if cutoff_time_column and config.use_cutoff_time:
            cutoff_times = target_df_copy[[target_index, cutoff_time_column]].copy()
            cutoff_times.columns = ["instance_id", "time"]
        
        # Compute feature matrix using featuretools
        with tqdm.tqdm(total=100) as pbar:
            
            def _cb(update, progress_percent, time_elapsed):
                pbar.update(int(update))
            
            feature_matrix = ft.calculate_feature_matrix(
                features=features,
                entityset=entity_set,
                cutoff_time=cutoff_times,
                chunk_size=config.chunk_size,
                n_jobs=config.n_jobs,
                progress_callback=_cb
            )
        
        # Ensure the feature matrix is aligned with original target dataframe order
        if target_index in feature_matrix.index.names or target_index == feature_matrix.index.name:
            # Reset index to get target_index as a column
            feature_matrix = feature_matrix.reset_index()
        elif feature_matrix.index.name is not None and feature_matrix.index.name != target_index:
            # Reset index but drop the unwanted index column
            feature_matrix = feature_matrix.reset_index(drop=True)
            # Add the target index column back
            feature_matrix[target_index] = range(len(feature_matrix))
        elif target_index not in feature_matrix.columns:
            # Add the target index column
            feature_matrix[target_index] = range(len(feature_matrix))
        
        # Merge with original target dataframe to preserve original columns and order
        original_target_with_index = target_dataframe.copy()
        if target_index not in original_target_with_index.columns:
            # Re-create the index column if it was synthetic
            original_target_with_index[target_index] = self._determine_target_index(original_target_with_index, key_mappings)
        
        # Ensure index columns have compatible types for merging
        if target_index in feature_matrix.columns and target_index in original_target_with_index.columns:
            # Convert both to integer for consistent merging (since we use range index)
            feature_matrix[target_index] = feature_matrix[target_index].astype(int)
            original_target_with_index[target_index] = original_target_with_index[target_index].astype(int)
        
        # Merge to get original columns + new features
        result = pd.merge(
            original_target_with_index,
            feature_matrix,
            on=target_index,
            how='left'
        )
        
        result = result.drop(columns=[target_index])
        
        return result
