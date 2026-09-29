"""Warm DFS session: plan + RDB ingest once, recompute per target split."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..dataset.meta import RDBColumnDType
from ..dataset.rdb import RDB
from .base_engine import DFSConfig, dfs_feature_column_name, get_dfs_engine
from .dfs2sql_engine import DFS2SQLEngine
from .duckdb_database import DuckDBBuilder

__all__ = ["DFSSession", "DFSSessionError", "create_dfs_session"]


class DFSSessionError(RuntimeError):
    """Raised when a DFSSession is used incorrectly (closed, wrong engine, etc.)."""


class DFSSession:
    """Reuse Featuretools plan + DuckDB RDB tables across train/val/test targets.

    First ``compute`` (or ``prepare``) freezes the feature list, SQL, and RDB
    ingest. Later ``compute`` calls only replace ``__target__`` / cutoff tables,
    re-execute SQL, and assemble. Column order is fixed after the first success.
    """

    def __init__(
        self,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config: Optional[DFSConfig] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = DFSConfig()
        effective = config.copy(deep=True)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(effective, key):
                    setattr(effective, key, value)

        if effective.engine != "dfs2sql":
            raise DFSSessionError(
                f"DFSSession currently requires engine='dfs2sql' (got {effective.engine!r})."
            )

        # Match compute_dfs_features key handling before any work.
        rdb = rdb.canonicalize_key_types()
        rdb.validate_key_consistency()

        self._rdb = rdb
        self._rdb_id = id(rdb)
        self._key_mappings = dict(key_mappings)
        self._cutoff_time_column = cutoff_time_column
        self._config = effective
        self._engine: DFS2SQLEngine = get_dfs_engine("dfs2sql", effective)  # type: ignore[assignment]

        self._builder: Optional[DuckDBBuilder] = None
        self._engine_path: Optional[str] = None
        self._features: Optional[List[Any]] = None
        self._sqls: Optional[List[Any]] = None
        self._sql_column_order: Optional[List[str]] = None
        self._cutoff_time_col_name: Optional[str] = None
        self._pinned_dtypes: Dict[str, Any] = {}
        self._feature_renames: Dict[str, str] = {}
        self.feature_columns_: Optional[List[str]] = None
        self.n_features_: int = 0

        self._prepared = False
        self._closed = False

        # Counters for tests / diagnostics
        self.n_plan_calls = 0
        self.n_rdb_ingest_calls = 0
        self.n_target_ingest_calls = 0
        self.n_sql_exec_calls = 0

    def __enter__(self) -> "DFSSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release DuckDB connection and delete temp DB file when applicable."""
        if self._closed:
            return
        self._closed = True
        if self._builder is not None:
            try:
                self._builder.db.close()
            except Exception as exc:
                logger.debug("DFSSession: error closing DuckDB: {}", exc)
            self._builder = None
        path = self._engine_path
        if path and path != ":memory:" and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                logger.debug("DFSSession: could not remove db file {}: {}", path, exc)
        self._engine_path = None

    def prepare(self, target_schema: pd.DataFrame) -> None:
        """Plan features and ingest RDB (+ schema target) without returning a matrix.

        ``target_schema`` must include key / cutoff columns; values may be empty or a
        small sample — used for dtypes and Featuretools EntitySet target registration.
        """
        self._ensure_open()
        if self._prepared:
            return
        prepared_target = self._prepare_target_frame(target_schema, pin_dtypes=True)
        self._plan_and_ingest_rdb(prepared_target)

    def compute(self, target_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute DFS features for one target frame, reusing warm state when available."""
        self._ensure_open()
        if len(target_dataframe) == 0:
            return target_dataframe

        original_index = target_dataframe.index
        # Defensive copy so key coercion does not mutate caller unexpectedly across splits.
        target_dataframe = target_dataframe.copy()
        self._coerce_key_columns(target_dataframe)

        target_index = "__target_index__"
        if target_index in target_dataframe.columns:
            raise DFSSessionError(
                f"Target dataframe cannot contain reserved column name '{target_index}'."
            )

        target_df_with_index = target_dataframe.assign(
            **{target_index: np.arange(len(target_dataframe))}
        )
        columns_to_keep = {target_index}
        columns_to_keep.update(self._key_mappings.keys())
        if self._cutoff_time_column:
            columns_to_keep.add(self._cutoff_time_column)
        target_df_for_engine = target_df_with_index[list(columns_to_keep)].copy()
        target_df_for_engine = self._apply_pinned_dtypes(target_df_for_engine)

        if not self._prepared:
            self._plan_and_ingest_rdb(target_df_for_engine)

        assert self._builder is not None
        assert self._features is not None
        assert self._sqls is not None
        assert self._sql_column_order is not None

        replace_target = self.n_target_ingest_calls > 0
        self._engine.ingest_or_replace_target(
            self._builder,
            target_df_for_engine,
            target_index,
            self._cutoff_time_column,
            replace=replace_target,
        )
        self.n_target_ingest_calls += 1

        feature_matrix = self._engine.execute_and_assemble(
            self._sqls,
            db=self._builder.db,
            target_dataframe=target_df_for_engine,
            column_order=self._sql_column_order,
            config=self._config,
            cutoff_time_col_name=self._cutoff_time_col_name,
        )
        self.n_sql_exec_calls += 1

        if self._feature_renames:
            feature_matrix = feature_matrix.rename(columns=self._feature_renames, copy=False)

        if self.feature_columns_ is None:
            # Stable output feature column order (after renames), excluding target index.
            self.feature_columns_ = [
                c for c in feature_matrix.columns if c != target_index
            ]
            self.n_features_ = len(self.feature_columns_)
        else:
            missing = [c for c in self.feature_columns_ if c not in feature_matrix.columns]
            if missing:
                raise DFSSessionError(
                    f"Session compute missing expected feature columns: {missing[:5]}…"
                    if len(missing) > 5
                    else f"Session compute missing expected feature columns: {missing}"
                )
            feature_matrix = feature_matrix[[target_index] + self.feature_columns_]

        if target_index not in feature_matrix.columns:
            raise DFSSessionError("Feature matrix is missing '__target_index__'.")

        target_with_idx = target_df_with_index.set_index(target_index)
        features_with_idx = feature_matrix.set_index(target_index)
        merged = target_with_idx.join(features_with_idx, how="left")
        merged.index = original_index
        return merged

    # ------------------------------------------------------------------ internals

    def _ensure_open(self) -> None:
        if self._closed:
            raise DFSSessionError("DFSSession is closed.")

    def _coerce_key_columns(self, target_dataframe: pd.DataFrame) -> None:
        for target_col, rdb_key in self._key_mappings.items():
            if target_col not in target_dataframe.columns:
                raise ValueError(f"Key column '{target_col}' not found in target dataframe.")
            table_name, col_name = rdb_key.split(".")
            try:
                table_meta = self._rdb.get_table_metadata(table_name)
            except ValueError as exc:
                raise ValueError(f"Table '{table_name}' not found in RDB.") from exc
            if col_name not in table_meta.column_dict:
                raise ValueError(f"Column '{col_name}' not found in table '{table_name}'.")
            col_meta = table_meta.column_dict[col_name]
            if col_meta.dtype != RDBColumnDType.primary_key:
                raise ValueError(
                    f"RDB column '{rdb_key}' is not a primary key. "
                    "Key mappings must point to primary keys."
                )
            if col_meta.dtype in (RDBColumnDType.primary_key, RDBColumnDType.foreign_key):
                target_col_dtype = target_dataframe[target_col].dtype
                if not pd.api.types.is_string_dtype(target_col_dtype):
                    target_dataframe[target_col] = target_dataframe[target_col].astype(str)

    def _prepare_target_frame(
        self, target_dataframe: pd.DataFrame, *, pin_dtypes: bool
    ) -> pd.DataFrame:
        if len(target_dataframe) == 0 and "__target_index__" not in target_dataframe.columns:
            # Allow empty schema frames for prepare(): still need columns present.
            frame = target_dataframe.copy()
            self._coerce_key_columns(frame)
            frame = frame.assign(**{"__target_index__": pd.Series(dtype=np.int64)})
            columns_to_keep = {"__target_index__"}
            columns_to_keep.update(self._key_mappings.keys())
            if self._cutoff_time_column:
                columns_to_keep.add(self._cutoff_time_column)
            frame = frame[list(columns_to_keep)]
            if pin_dtypes:
                self._pin_dtypes_from(frame)
            return self._apply_pinned_dtypes(frame)

        # Non-empty: mirror compute() prep without full matrix path.
        frame = target_dataframe.copy()
        self._coerce_key_columns(frame)
        target_index = "__target_index__"
        frame = frame.assign(**{target_index: np.arange(len(frame))})
        columns_to_keep = {target_index}
        columns_to_keep.update(self._key_mappings.keys())
        if self._cutoff_time_column:
            columns_to_keep.add(self._cutoff_time_column)
        frame = frame[list(columns_to_keep)].copy()
        if pin_dtypes:
            self._pin_dtypes_from(frame)
        return self._apply_pinned_dtypes(frame)

    def _pin_dtypes_from(self, target_df: pd.DataFrame) -> None:
        cols = list(self._key_mappings.keys())
        if self._cutoff_time_column:
            cols.append(self._cutoff_time_column)
        self._pinned_dtypes = {c: target_df[c].dtype for c in cols if c in target_df.columns}

    def _apply_pinned_dtypes(self, target_df: pd.DataFrame) -> pd.DataFrame:
        if not self._pinned_dtypes:
            self._pin_dtypes_from(target_df)
            return target_df
        out = target_df
        for col, dtype in self._pinned_dtypes.items():
            if col not in out.columns:
                continue
            if out[col].dtype != dtype:
                try:
                    out[col] = out[col].astype(dtype)
                except (TypeError, ValueError) as exc:
                    raise DFSSessionError(
                        f"Failed to coerce target column '{col}' to pinned dtype {dtype}: {exc}"
                    ) from exc
        return out

    def _plan_and_ingest_rdb(self, target_df_for_engine: pd.DataFrame) -> None:
        if id(self._rdb) != self._rdb_id:
            raise DFSSessionError("RDB object identity changed after session creation.")

        features = self._engine.prepare_features(
            self._rdb,
            target_df_for_engine,
            self._key_mappings,
            self._cutoff_time_column,
            self._config,
        )
        self.n_plan_calls += 1
        self._features = features
        logger.info(
            "DFSSession: planned {} features (prepare call #{})",
            len(features),
            self.n_plan_calls,
        )
        if not features:
            raise DFSSessionError("No features generated; check configuration or data.")

        renames: Dict[str, str] = {}
        for f in features:
            old = f.get_name()
            new = dfs_feature_column_name(f)
            if old != new:
                renames[old] = new
        self._feature_renames = renames

        builder, engine_path = self._engine.open_builder(self._config)
        self._builder = builder
        self._engine_path = engine_path
        self._engine.ingest_rdb_tables(builder, self._rdb)
        self.n_rdb_ingest_calls += 1

        # features2sql needs cutoff / time-index *names*; table contents are loaded in compute().
        if self._config.use_cutoff_time and self._cutoff_time_column is not None:
            from ..dataset.meta import RDBCutoffTime

            builder.cutoff_time_col_name = RDBCutoffTime.column_name.value
            builder.cutoff_time_table_name = RDBCutoffTime.table_name.value
            builder.time_columns["__target__"] = self._cutoff_time_column

        sqls, column_order, cutoff_time_col_name = self._engine.plan_feature_sqls(
            builder,
            self._rdb,
            target_df_for_engine,
            features,
            self._cutoff_time_column,
            self._config,
        )
        self._sqls = sqls
        self._sql_column_order = column_order
        self._cutoff_time_col_name = cutoff_time_col_name
        self._prepared = True


def create_dfs_session(
    rdb: RDB,
    key_mappings: Dict[str, str],
    cutoff_time_column: Optional[str] = None,
    config: Optional[DFSConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> DFSSession:
    """Create a warm DFS session bound to one RDB and key/cutoff config."""
    return DFSSession(
        rdb=rdb,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff_time_column,
        config=config,
        config_overrides=config_overrides,
    )
