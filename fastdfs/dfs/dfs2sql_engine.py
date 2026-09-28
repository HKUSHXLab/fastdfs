"""
SQL-based DFS engine implementation for the new interface.
"""

import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import featuretools as ft
import pandas as pd
import tqdm
from loguru import logger
from sql_formatter.core import format_sql

from ..dataset.meta import RDBColumnDType, RDBCutoffTime
from ..dataset.rdb import RDB
from .base_engine import DFSConfig, DFSEngine, dfs_engine
from .duckdb_database import DuckDBBuilder
from .gen_sqls import decode_column_from_sql, features2sql
from .merge_sql import merge_queries

__all__ = ['DFS2SQLEngine', 'assemble_dfs2sql_feature_frames']


def merge_dfs2sql_feature_frames_legacy(
    dataframes: List[pd.DataFrame],
    target_index: str,
) -> pd.DataFrame:
    """Sequential left merges (original dfs2sql behavior). Used in tests for parity checks."""
    if not dataframes:
        raise ValueError("merge_dfs2sql_feature_frames_legacy: empty dataframes")
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=target_index, how="left")
    return merged_df.sort_values(by=target_index).reset_index(drop=True)


def assemble_dfs2sql_feature_frames(
    dataframes: List[pd.DataFrame],
    target_index: str,
    canonical_index: pd.Index,
    *,
    concat_chunk_size: int = 512,
) -> pd.DataFrame:
    """Horizontally stitch per-feature SQL frames aligned on ``target_index``.

    Replaces O(N) sequential ``pd.merge`` calls with index-aligned ``pd.concat``
    (chunked along axis=1 for large N to cap intermediate width growth patterns).

    Args:
        dataframes: One DataFrame per successful DuckDB query; each includes ``target_index``
            and one or more feature columns.
        target_index: Name of the synthetic target key column (e.g. ``__target_index__``).
        canonical_index: Row order for the output; typically ``target_dataframe[target_index]``.
        concat_chunk_size: Number of skinny frames to concat per intermediate block.

    Returns:
        Wide frame sorted by ``target_index``, same contract as the legacy merge path.
    """
    if not canonical_index.is_unique:
        raise ValueError(
            f"assemble_dfs2sql_feature_frames: {target_index!r} must be unique in the target frame."
        )
    if not dataframes:
        return pd.DataFrame({target_index: canonical_index.to_numpy()})

    aligned: List[pd.DataFrame] = []
    seen_cols: set[str] = set()
    for i, df in enumerate(dataframes):
        if target_index not in df.columns:
            raise ValueError(
                f"assemble_dfs2sql_feature_frames: frame {i} missing column {target_index!r}."
            )
        part = df.drop_duplicates(subset=[target_index], keep="first").set_index(target_index)
        dup_names = set(part.columns) & seen_cols
        if dup_names:
            raise ValueError(
                f"assemble_dfs2sql_feature_frames: duplicate feature column(s) {sorted(dup_names)!r}."
            )
        seen_cols.update(part.columns)
        aligned.append(part.reindex(canonical_index))

    k = max(1, int(concat_chunk_size))
    if len(aligned) == 1:
        wide = aligned[0]
    else:
        blocks: List[pd.DataFrame] = []
        for j in range(0, len(aligned), k):
            blocks.append(pd.concat(aligned[j : j + k], axis=1))
        wide = pd.concat(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    wide.index.name = target_index
    out = wide.reset_index()
    return out.sort_values(by=target_index).reset_index(drop=True)


@dfs_engine
class DFS2SQLEngine(DFSEngine):
    """SQL-based DFS engine implementation."""

    name = "dfs2sql"

    def compute_feature_matrix(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using SQL generation (reuse existing computation logic)."""
        # Set up database with RDB tables + target table
        target_index = "__target_index__"  # Target index is already handled by base class
        
        engine_path = config.engine_path
        if engine_path is None:
            # Generate a random temporary file path if not specified
            engine_path = str(Path(tempfile.gettempdir()) / f"fastdfs_{uuid.uuid4()}.db")
            logger.debug(f"Using temporary DuckDB path: {engine_path}")
            
        builder = DuckDBBuilder(Path(engine_path))
        for setting, value in (
            ("threads", config.dfs2sql_threads),
            ("memory_limit", config.dfs2sql_memory_limit),
            ("temp_directory", config.dfs2sql_temp_directory),
        ):
            if value is not None:
                builder.db.execute(f"SET {setting} = ?", [value])
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
            include_cutoff_time=config.include_cutoff_time,
        )

        column_order = list(
            dict.fromkeys(
                decode_column_from_sql(name)
                for query in sqls
                for name in query.named_selects
                if name != cutoff_time_col_name
            )
        )
        if config.dfs2sql_merge_queries:
            sqls = merge_queries(sqls)
        logger.debug("Executing SQLs ...")
        dataframes = self._execute_feature_sqls(
            sqls,
            db=db,
            config=config,
            cutoff_time_col_name=cutoff_time_col_name,
        )

        # Assemble all feature dataframes (index-aligned concat; see ``assemble_dfs2sql_feature_frames``).
        if dataframes:
            canonical_index = pd.Index(target_dataframe[target_index].values, name=target_index)
            t0 = time.perf_counter()
            logger.info(
                "dfs2sql: assembling {} feature frame(s) via concat (chunk_size={}) …",
                len(dataframes),
                config.dfs2sql_concat_chunk_size,
            )
            merged_df = assemble_dfs2sql_feature_frames(
                dataframes,
                target_index,
                canonical_index,
                concat_chunk_size=config.dfs2sql_concat_chunk_size,
            )
            logger.info("dfs2sql: assembly finished in {:.2f}s", time.perf_counter() - t0)

            columns_to_exclude = set(target_dataframe.columns) - {target_index}
            feature_columns = [
                col
                for col in column_order
                if col in merged_df.columns and col not in columns_to_exclude
            ]

            return merged_df[feature_columns]
        else:
            # No features generated
            logger.warning("No features generated from SQL execution.")
            # Return dataframe with just the target index to satisfy contract
            return pd.DataFrame({target_index: target_dataframe[target_index]})

    def _effective_sql_workers(self, config: DFSConfig) -> int:
        return max(1, int(getattr(config, "dfs2sql_sql_workers", 1) or 1))

    @staticmethod
    def _postprocess_sql_frame(
        dataframe: pd.DataFrame,
        cutoff_time_col_name: Optional[str],
    ) -> pd.DataFrame:
        if cutoff_time_col_name is not None and cutoff_time_col_name in dataframe.columns:
            dataframe = dataframe.drop(columns=[cutoff_time_col_name])
        return dataframe.rename(decode_column_from_sql, axis="columns")

    def _execute_feature_sqls(
        self,
        sqls: List[Any],
        *,
        db: duckdb.DuckDBPyConnection,
        config: DFSConfig,
        cutoff_time_col_name: Optional[str],
    ) -> List[pd.DataFrame]:
        """Run concurrent queries through cursors sharing one database."""
        workers = self._effective_sql_workers(config)
        sql_texts = [sql.sql() for sql in sqls]
        if workers == 1 or len(sql_texts) <= 1:
            dataframes: List[pd.DataFrame] = []
            for text in tqdm.tqdm(sql_texts):
                logger.debug("Executing SQL: {}", format_sql(text))
                result = db.sql(text)
                if result is not None:
                    dataframes.append(
                        self._postprocess_sql_frame(result.df(), cutoff_time_col_name)
                    )
            return dataframes

        logger.info(
            "dfs2sql: executing {} SQL(s) with up to {} concurrent queries",
            len(sql_texts),
            workers,
        )
        def _run_one(item: Tuple[int, str]) -> Tuple[int, Optional[pd.DataFrame]]:
            idx, text = item
            logger.debug("Executing SQL[{}]: {}", idx, format_sql(text))
            # A cursor shares the database even when it has no backing file.
            connection = db.cursor()
            try:
                result = connection.sql(text)
                frame = (
                    None
                    if result is None
                    else self._postprocess_sql_frame(result.df(), cutoff_time_col_name)
                )
                return idx, frame
            finally:
                connection.close()

        slots: List[Optional[pd.DataFrame]] = [None] * len(sql_texts)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_one, (i, text)) for i, text in enumerate(sql_texts)
            ]
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
                idx, frame = fut.result()
                slots[idx] = frame

        return [frame for frame in slots if frame is not None]

    def _build_database_tables(
        self,
        builder: DuckDBBuilder,
        rdb: RDB,
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

    def _build_column_type_map(self, rdb: RDB, target_dataframe: pd.DataFrame) -> Dict[Tuple[str, str], str]:
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
