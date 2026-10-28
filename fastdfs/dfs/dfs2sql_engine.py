"""
SQL-based DFS engine implementation for the new interface.
"""

import os
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
import woodwork as ww
from loguru import logger
from sql_formatter.core import format_sql

from ..dataset.meta import RDBColumnDType, RDBCutoffTime
from ..dataset.rdb import RDB
from .base_engine import DFSConfig, DFSEngine, dfs_engine, is_valid_time_column, to_naive_datetime
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


def release_duckdb(db: duckdb.DuckDBPyConnection, temp_db_path: Optional[str]) -> None:
    """Close a dfs2sql DuckDB connection and delete its generated temp DB file, if any.

    ``temp_db_path`` must be set only for a path fastdfs generated itself; a
    user-supplied ``engine_path`` is never deleted. Errors are logged, not raised.
    """
    try:
        db.close()
    except Exception as exc:
        logger.debug(f"Error closing DuckDB: {exc}")
    if temp_db_path and os.path.exists(temp_db_path):
        try:
            os.remove(temp_db_path)
        except OSError as exc:
            logger.debug(f"Could not remove temp DuckDB file {temp_db_path}: {exc}")


def _first_non_float(series: pd.Series) -> Any:
    """First value float() rejects (for error messages; woodwork's Double casts with float())."""
    for value in series.dropna().unique():
        try:
            float(value)
        except (TypeError, ValueError):
            return value
    return None


@dfs_engine
class DFS2SQLEngine(DFSEngine):
    """SQL-based DFS engine implementation."""

    name = "dfs2sql"
    plans_on_empty_frames = True

    def open_builder(self, config: DFSConfig) -> Tuple[DuckDBBuilder, str]:
        """Create a DuckDB builder/connection for dfs2sql (caller owns lifecycle)."""
        engine_path = config.engine_path
        if engine_path is None:
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
        return builder, engine_path

    def ingest_rdb_tables(self, builder: DuckDBBuilder, rdb: RDB) -> None:
        """Load all RDB tables into DuckDB (once per warm session).

        This is the only place that sets the column types DuckDB computes on. Types
        come from RDB metadata, and values are converted with woodwork's logical
        types, the same conversions (and checks) full-frame planning applied:

        - ``float`` -> woodwork ``Double`` -> ``float64`` (DOUBLE), also for bool, int
          and float32 columns. Values that cannot be converted (e.g. ``"x"``) raise a
          ``ValueError``; strings woodwork treats as missing (``"nan"``, ...) are allowed.
        - ``category`` -> pandas ``category`` (ENUM).
        - ``datetime`` -> woodwork ``Datetime`` -> naive datetime (TIMESTAMP_NS).
          Values that don't fit become NaT with a warning, except in the table's time
          column, which must be numeric or parse as datetimes (``ValueError``).
        - ``timestamp`` -> numeric.
        - Primary keys must be unique and non-null; an ``IndexError`` is raised otherwise.
        - ``text`` and primary/foreign keys are loaded as they are (keys are strings,
          so VARCHAR). Keys are only join columns, never feature values, and
          converting high-cardinality keys to category is expensive. One visible
          effect: DuckDB's ``mode()`` has no fixed tie-break, so a ``MODE`` feature
          with tied counts may pick a different value than full Featuretools
          planning did (which turned keys into ENUM).
        """
        for table_name in rdb.table_names:
            # Shallow copy: converted columns replace columns in the copy only, so the
            # caller's RDB DataFrames are never modified (data is not duplicated).
            df = rdb.get_table(table_name).copy(deep=False)
            table_meta = rdb.get_table_metadata(table_name)

            for col_schema in table_meta.columns:
                if col_schema.name not in df.columns:
                    continue
                original = df[col_schema.name]
                # dtype is an RDBColumnDType, or a plain string for some loaded metadata.
                where = f"'{table_name}.{col_schema.name}' ({RDBColumnDType(col_schema.dtype).value})"
                if col_schema.dtype == RDBColumnDType.float_t:
                    try:
                        df[col_schema.name] = ww.logical_types.Double().transform(original)
                    except (TypeError, ValueError, ww.exceptions.TypeConversionError) as exc:
                        raise ValueError(
                            f"Column {where} has values that cannot be converted to float, "
                            f"e.g. {_first_non_float(original)!r}"
                        ) from exc
                elif col_schema.dtype == RDBColumnDType.datetime_t:
                    if col_schema.name == table_meta.time_column and not is_valid_time_column(original):
                        raise ValueError(
                            f"Time column {where} must contain datetime or numeric values; "
                            f"some values cannot be parsed as datetimes."
                        )
                    df[col_schema.name] = to_naive_datetime(original)
                elif col_schema.dtype == RDBColumnDType.timestamp_t:
                    df[col_schema.name] = pd.to_numeric(original, errors='coerce')
                elif col_schema.dtype == RDBColumnDType.category_t:
                    if not isinstance(original.dtype, pd.CategoricalDtype):
                        df[col_schema.name] = original.astype("category")

            index_col = self._get_table_index(table_meta)
            if index_col == "__index__" and "__index__" not in df.columns:
                df["__index__"] = range(len(df))
            else:
                # Planning sees 0-row frames, so woodwork no longer validates keys.
                # Same checks (and messages) as woodwork's index check on master.
                keys = df[index_col]
                if not keys.is_unique:
                    raise IndexError(
                        f"Index column must be unique: primary key '{table_name}.{index_col}' "
                        f"has duplicate values, e.g. {keys[keys.duplicated()].iloc[0]!r}"
                    )
                if keys.isnull().any():
                    raise IndexError(
                        f"Index contains null values: primary key '{table_name}.{index_col}'"
                    )

            builder.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=index_col,
                time_index=table_meta.time_column,
            )

    def ingest_or_replace_target(
        self,
        builder: DuckDBBuilder,
        target_dataframe: pd.DataFrame,
        target_index: str,
        cutoff_time_column: Optional[str],
        key_columns: List[str],
        *,
        replace: bool = False,
    ) -> None:
        """Load or replace ``__target__`` (+ cutoff helper) for the current split.

        Key columns are cast to strings, so every split loads them as VARCHAR, even
        when it is empty or its keys are all missing (an ``object`` column with no
        strings would otherwise load as INTEGER). The cutoff column is already
        ``datetime64[ns]`` (or numeric) from ``_prepare_target_for_engine``.
        """
        target_dataframe = target_dataframe.copy(deep=False)
        for key_col in key_columns:
            target_dataframe[key_col] = target_dataframe[key_col].astype("string")
        builder.add_dataframe(
            dataframe_name="__target__",
            dataframe=target_dataframe,
            index=target_index,
            time_index=cutoff_time_column,
            replace=replace,
        )
        builder.index_name = target_index
        builder.index = target_dataframe[target_index].values

        if cutoff_time_column:
            cutoff_time = target_dataframe[[target_index, cutoff_time_column]].copy()
            cutoff_time.columns = [target_index, RDBCutoffTime.column_name.value]
            builder.set_cutoff_time(cutoff_time, replace=replace)

    def plan_feature_sqls(
        self,
        builder: DuckDBBuilder,
        features: List[ft.FeatureBase],
        cutoff_time_column: Optional[str],
        config: DFSConfig,
    ) -> Tuple[List[Any], List[str], Optional[str]]:
        """Compile Featuretools features to (optionally merged) SQL queries.

        Must run after the RDB tables and the target are ingested, because column
        types are read from DuckDB.
        """
        target_index = "__target_index__"
        has_cutoff_time = config.use_cutoff_time and cutoff_time_column is not None
        if has_cutoff_time:
            time_columns = builder.time_columns
            cutoff_time_table_name = builder.cutoff_time_table_name
            cutoff_time_col_name = builder.cutoff_time_col_name
        else:
            time_columns = None
            cutoff_time_table_name = None
            cutoff_time_col_name = None

        column_type_map = self._build_column_type_map(builder.db)
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
        return sqls, column_order, cutoff_time_col_name

    def execute_and_assemble(
        self,
        sqls: List[Any],
        *,
        db: duckdb.DuckDBPyConnection,
        target_dataframe: pd.DataFrame,
        column_order: List[str],
        config: DFSConfig,
        cutoff_time_col_name: Optional[str],
    ) -> pd.DataFrame:
        """Run frozen SQLs and stitch results into a feature matrix keyed by target index."""
        target_index = "__target_index__"
        logger.debug("Executing SQLs ...")
        dataframes = self._execute_feature_sqls(
            sqls,
            db=db,
            config=config,
            cutoff_time_col_name=cutoff_time_col_name,
        )
        if not dataframes:
            logger.warning("No features generated from SQL execution.")
            return pd.DataFrame({target_index: target_dataframe[target_index]})

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

    def compute_feature_matrix(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Compute feature values using SQL generation (one-shot; opens a fresh DuckDB)."""
        target_index = "__target_index__"
        builder, engine_path = self.open_builder(config)
        try:
            self.ingest_rdb_tables(builder, rdb)
            self.ingest_or_replace_target(
                builder, target_dataframe, target_index, cutoff_time_column, list(key_mappings)
            )
            sqls, column_order, cutoff_time_col_name = self.plan_feature_sqls(
                builder, features, cutoff_time_column, config
            )
            return self.execute_and_assemble(
                sqls,
                db=builder.db,
                target_dataframe=target_dataframe,
                column_order=column_order,
                config=config,
                cutoff_time_col_name=cutoff_time_col_name,
            )
        finally:
            # open_builder generated the path only when engine_path is None.
            release_duckdb(builder.db, engine_path if config.engine_path is None else None)

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

    def _get_table_index(self, table_meta) -> str:
        """Get the primary key column for a table."""
        for col_schema in table_meta.columns:
            if col_schema.dtype == 'primary_key':
                return col_schema.name
        # If no primary key, create default index
        return "__index__"

    def _build_column_type_map(self, db: duckdb.DuckDBPyConnection) -> Dict[Tuple[str, str], str]:
        """Map (table_name, column_name) to the DuckDB type the SQL will run on.

        Read from DuckDB itself after ingest, so SQL generation (e.g. casting
        BOOLEAN for numeric aggregations) always matches the loaded data.
        """
        rows = db.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = current_database() AND table_schema = current_schema()"
        ).fetchall()
        return {(table, column): data_type for table, column, data_type in rows}
