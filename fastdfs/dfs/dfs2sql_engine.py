"""
SQL-based DFS engine implementation for the new interface.
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import featuretools as ft
from sql_formatter.core import format_sql
import tqdm
from loguru import logger

from .base_engine import DFSEngine, DFSConfig, dfs_engine
from ..dataset.rdb import RDBDataset
from .gen_sqls import features2sql, decode_column_from_sql
from .duckdb_database import DuckDBBuilder
from ..dataset.meta import RDBCutoffTime, RDBColumnDType

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
        target_index = "__target_index__"  # Target index is already handled by base class
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

        # Build column type map from RDB tables for boolean detection
        column_type_map = self._build_column_type_map(rdb, target_dataframe)

        sqls = features2sql(
            features,
            target_index,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name=cutoff_time_table_name,
            cutoff_time_col_name=cutoff_time_col_name,
            time_col_mapping=time_columns,
            column_type_map=column_type_map,
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
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = pd.merge(merged_df, df, on=target_index, how='left')

            merged_df = merged_df.sort_values(by=target_index).reset_index(drop=True)

            columns_to_exclude = set(target_dataframe.columns) - {target_index}
            feature_columns = [col for col in merged_df.columns if col not in columns_to_exclude]

            return merged_df[feature_columns]
        else:
            # No features generated
            logger.warning("No features generated from SQL execution.")
            return None

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

            # Enforce types based on metadata to avoid DuckDB inferring VARCHAR for numeric columns
            for col_schema in table_meta.columns:
                if col_schema.name in df.columns:
                    if col_schema.dtype == RDBColumnDType.float_t:
                        df[col_schema.name] = pd.to_numeric(df[col_schema.name], errors='coerce')
                    elif col_schema.dtype == RDBColumnDType.datetime_t:
                        df[col_schema.name] = pd.to_datetime(df[col_schema.name], errors='coerce')
                    elif col_schema.dtype == RDBColumnDType.timestamp_t:
                        df[col_schema.name] = pd.to_numeric(df[col_schema.name], errors='coerce')

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

        # Add target dataframe as __target__ table (target_index is already in dataframe)
        target_df_for_db = target_dataframe

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
            # Create cutoff time dataframe with only necessary columns
            cutoff_time = target_df_for_db[[target_index, cutoff_time_column]]
            cutoff_time.columns = [target_index, RDBCutoffTime.column_name.value]
            builder.set_cutoff_time(cutoff_time)

    def _get_table_index(self, table_meta) -> str:
        """Get the primary key column for a table."""
        for col_schema in table_meta.columns:
            if col_schema.dtype == 'primary_key':
                return col_schema.name
        # If no primary key, create default index
        return "__index__"

    def _build_column_type_map(self, rdb: RDBDataset, target_dataframe: pd.DataFrame) -> Dict[Tuple[str, str], str]:
        """Build a mapping from (table_name, column_name) to dtype string for boolean detection.
        
        Returns:
            Dictionary mapping (table_name, column_name) tuples to dtype strings
        """
        column_type_map = {}
        
        # Add RDB table columns
        for table_name in rdb.table_names:
            df = rdb.get_table(table_name)
            for col_name in df.columns:
                dtype_str = str(df[col_name].dtype)
                column_type_map[(table_name, col_name)] = dtype_str
        
        # Add target dataframe columns
        for col_name in target_dataframe.columns:
            dtype_str = str(target_dataframe[col_name].dtype)
            column_type_map[("__target__", col_name)] = dtype_str
        
        return column_type_map