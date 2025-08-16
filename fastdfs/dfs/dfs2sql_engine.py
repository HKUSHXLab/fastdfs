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

__all__ = ['DFS2SQLEngine']


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

        # Set up database with RDB tables + target table
        target_index = self._determine_target_index(target_dataframe, key_mappings)
        builder = DuckDBBuilder(Path(config.engine_path))
        self._build_database_tables(builder, rdb, target_dataframe, target_index, cutoff_time_column)
        db = builder.db

        # Generate SQLs from feature specifications (reuse existing features2sql logic)
        has_cutoff_time = config.use_cutoff_time and cutoff_time_column is not None
        if has_cutoff_time:
            time_columns = builder.time_columns
            cutoff_time_table_name = builder.cutoff_time_table_name
            cutoff_time_col_name = builder.cutoff_time_col_name
        else:
            time_columns = None
            cutoff_time_table_name = None
            cutoff_time_col_name = None

        sqls = features2sql(
            features,
            target_index,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name=cutoff_time_table_name,
            cutoff_time_col_name=cutoff_time_col_name,
            time_col_mapping=time_columns,
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
                if cutoff_time_col_name in dataframe.columns:
                    dataframe.drop(columns=[cutoff_time_col_name], inplace=True)
                dataframe.rename(decode_column_from_sql, axis="columns", inplace=True)
                dataframes.append(dataframe)
            else:
                logger.warning(f"SQL execution returned None for: {format_sql(sql.sql())}")

        # Merge all feature dataframes
        if dataframes:
            logger.debug("Finalizing ...")
            merged_df = pd.DataFrame(
                reduce(lambda left, right: pd.merge(left, right, on=target_index), dataframes)
            )

            # Merge with original target dataframe to preserve original columns and order
            # Shallow copy to allow adding synthetic index column without affecting original
            if target_index not in target_dataframe.columns:
                original_target_with_index = target_dataframe.copy(deep=False)
                original_target_with_index[target_index] = self._determine_target_index(original_target_with_index, key_mappings)
            else:
                original_target_with_index = target_dataframe

            # Merge to get original columns + new features
            result = pd.merge(
                original_target_with_index,
                merged_df,
                on=target_index,
                how='left'
            )

            # Remove the synthetic target index
            result = result.drop(columns=[target_index])

            return result
        else:
            return target_dataframe

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

            # Get the appropriate index column
            index_col = self._get_table_index(table_meta)

            # Add __index__ column if it doesn't have a primary key (shallow copy for new columns)
            if index_col == "__index__" and "__index__" not in df.columns:
                df = df.copy(deep=False)  # Shallow copy - shares data but allows new columns
                df["__index__"] = range(len(df))

            # Add table to database
            builder.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=index_col,
                time_index=table_meta.time_column
            )

        # Add target dataframe as __target__ table
        target_df_for_db = target_dataframe
        needs_index_column = target_index not in target_dataframe.columns
        
        if needs_index_column:
            # Shallow copy to allow adding synthetic index column without affecting original
            target_df_for_db = target_dataframe.copy(deep=False)
            target_df_for_db[target_index] = self._determine_target_index(target_df_for_db, {})

        builder.add_dataframe(
            dataframe_name="__target__",
            dataframe=target_df_for_db,
            index=target_index,
            time_index=cutoff_time_column
        )

        builder.index_name = target_index
        builder.index = target_df_for_db[target_index].values

        # Set up cutoff time information
        if cutoff_time_column:
            # Create cutoff time dataframe with only necessary columns (shallow copy)
            cutoff_time = target_df_for_db[[target_index, cutoff_time_column]].copy()
            cutoff_time.columns = [target_index, "time"]
            builder.set_cutoff_time(cutoff_time)

    def _get_table_index(self, table_meta) -> str:
        """Get the primary key column for a table."""
        for col_schema in table_meta.columns:
            if col_schema.dtype == 'primary_key':
                return col_schema.name
        # If no primary key, create default index
        return "__index__"