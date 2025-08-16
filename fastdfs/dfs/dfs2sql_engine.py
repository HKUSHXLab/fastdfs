"""
SQL-based DFS engine implementation for the new interface.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import featuretools as ft
from sql_formatter.core import format_sql
from functools import reduce
import tqdm
from loguru import logger

from .base_engine import DFSEngine, DFSConfig, dfs_engine
from ..dataset.rdb_simplified import RDBDataset
from ..preprocess.dfs.gen_sqls import features2sql, decode_column_from_sql
from ..preprocess.dfs.database import DuckDBBuilder
import numpy as np

__all__ = ['DFS2SQLEngine']


def _check_array_agg_occurrences(col_name, array_agg_func_names) -> int:
    """Count occurrences of array aggregation function names in column name."""
    arr_agg_counts = 0
    for func_name in array_agg_func_names:
        arr_agg_counts += col_name.count(func_name)
    return arr_agg_counts


def _nanstack(arr_list: List) -> np.ndarray:
    """Stack a list of numpy ndarrays that may contain NaN."""
    if arr_list is None:
        return np.nan  # all values are NaN
    arr_len = None
    for arr in arr_list:
        if isinstance(arr, list):
            arr_len = len(arr)
            break
    if arr_len is None:
        return np.nan  # all values are NaN
    fill_val = np.zeros(arr_len)
    new_arr_list = [arr if isinstance(arr, list) else fill_val for arr in arr_list]
    return np.stack(new_arr_list)


def array_max(column):
    """Apply max aggregation to array column."""
    if not isinstance(column, list):
        return np.nan
    stack = _nanstack(column)
    return stack.max(0) if isinstance(stack, np.ndarray) else np.nan


def array_min(column):
    """Apply min aggregation to array column."""
    if not isinstance(column, list):
        return np.nan
    stack = _nanstack(column)
    return stack.min(0) if isinstance(stack, np.ndarray) else np.nan


def array_mean(column):
    """Apply mean aggregation to array column."""
    if not isinstance(column, list):
        return np.nan
    stack = _nanstack(column)
    return stack.mean(0) if isinstance(stack, np.ndarray) else np.nan


@dfs_engine  
class DFS2SQLEngine(DFSEngine):
    """SQL-based DFS engine implementation."""
    
    name = "dfs2sql"
    
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using SQL generation (reuse existing computation logic)."""
        
        # Apply DFS2SQL-specific feature filtering
        filtered_features = self._filter_nested_array_agg_features(features)
        
        # Set up database with RDB tables + target table
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        builder = DuckDBBuilder(Path(config.engine_path))
        self._build_database_tables(builder, rdb, target_dataframe, target_index, cutoff_time_column)
        db = builder.db
        
        # Generate SQLs from feature specifications (reuse existing features2sql logic)
        has_cutoff_time = config.use_cutoff_time and cutoff_time_column is not None
        sqls = features2sql(
            filtered_features,
            target_index,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name="__target__",
            cutoff_time_col_name=cutoff_time_column,
            time_col_mapping=builder.time_columns
        )
        
        # Execute SQLs and merge results (reuse existing logic)
        logger.debug("Executing SQLs ...")
        dataframes = []
        for sql in tqdm.tqdm(sqls):
            logger.debug(f"Executing SQL: {format_sql(sql.sql())}")
            result = db.sql(sql.sql())
            if result is not None:
                dataframe = result.df()
                
                # Clean up result dataframe (reuse existing logic)
                if cutoff_time_column and cutoff_time_column in dataframe.columns:
                    dataframe.drop(columns=[cutoff_time_column], inplace=True)
                dataframe.rename(decode_column_from_sql, axis="columns", inplace=True)
                self._handle_array_aggregation(dataframe)
                dataframes.append(dataframe)
        
        # Merge all feature dataframes
        if dataframes:
            logger.debug("Finalizing ...")
            merged_df = pd.DataFrame(
                reduce(lambda left, right: pd.merge(left, right, on=target_index), dataframes)
            )
            
            # Merge with original target dataframe to preserve original columns and order
            original_target_with_index = target_dataframe.copy()
            if target_index not in original_target_with_index.columns:
                # Re-create the index column if it was synthetic
                original_target_with_index[target_index] = self._determine_target_index(original_target_with_index, key_mappings)
            
            # Merge to get original columns + new features
            result = pd.merge(
                original_target_with_index,
                merged_df,
                on=target_index,
                how='left'
            )
            
            # Remove the synthetic index column if it was added
            if len(key_mappings) > 1 and "_index" in target_index:
                result = result.drop(columns=[target_index])
            
            return result
        else:
            return target_dataframe.copy()
    
    def _filter_nested_array_agg_features(self, features: List[ft.FeatureBase]) -> List[ft.FeatureBase]:
        """Filter nested array aggregation features (reuse existing logic)."""
        if len(features) == 0:
            return features
            
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        new_features = []
        for feat in features:
            feat_str = str(feat)
            agg_count = _check_array_agg_occurrences(feat_str, array_agg_func_names)
            if agg_count > 1:
                # Remove features with nested array aggregation
                continue
            new_features.append(feat)
        return new_features
    
    def _build_database_tables(
        self, 
        builder: DuckDBBuilder, 
        rdb: RDBDataset, 
        target_dataframe: pd.DataFrame,
        target_index: str,
        cutoff_time_column: Optional[str]
    ):
        """Build database tables for SQL execution (adapted from existing build_dataframes logic)."""
        
        # Add all RDB tables to database
        for table_name in rdb.table_names:
            df = rdb.get_table(table_name)
            table_meta = rdb.get_table_metadata(table_name)
            
            # Add table to database
            builder.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=self._get_table_index(table_meta),
                time_index=table_meta.time_column
            )
        
        # Add target dataframe as __target__ table
        target_df_copy = target_dataframe.copy()
        if target_index not in target_df_copy.columns:
            # Re-create the index column if it was synthetic
            target_df_copy[target_index] = self._determine_target_index(target_df_copy, {})
        
        builder.add_dataframe(
            dataframe_name="__target__", 
            dataframe=target_df_copy,
            index=target_index,
            time_index=cutoff_time_column
        )
        
        builder.index_name = target_index
        builder.index = target_df_copy[target_index].values
        
        # Set up cutoff time information
        if cutoff_time_column:
            cutoff_time = target_df_copy[[target_index, cutoff_time_column]].copy()
            cutoff_time.columns = [target_index, "time"]
            builder.set_cutoff_time(cutoff_time)
    
    def _get_table_index(self, table_meta) -> str:
        """Get the primary key column for a table."""
        for col_schema in table_meta.columns:
            if col_schema.dtype == 'primary_key':
                return col_schema.name
        # If no primary key, create default index
        return "__index__"
    
    def _handle_array_aggregation(self, df: pd.DataFrame):
        """Handle array aggregation results (reuse existing logic)."""
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        for col in df.columns:
            num_array_agg = _check_array_agg_occurrences(col, array_agg_func_names)
            if num_array_agg == 1:
                if "ARRAYMAX" in col:
                    df[col] = df[col].apply(array_max)
                elif "ARRAYMIN" in col:
                    df[col] = df[col].apply(array_min)
                elif "ARRAYMEAN" in col:
                    df[col] = df[col].apply(array_mean)
