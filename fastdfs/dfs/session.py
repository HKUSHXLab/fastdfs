"""Warm DFS session: plan + RDB ingest once, recompute per target split."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from ..dataset.rdb import RDB
from .base_engine import (
    DFSConfig,
    coerce_target_key_types,
    dfs_feature_renames,
    get_dfs_engine,
)
from .dfs2sql_engine import DFS2SQLEngine
from .duckdb_database import DuckDBBuilder

__all__ = ["DFSSession", "DFSSessionError"]


class DFSSessionError(RuntimeError):
    """Raised when a DFSSession is used incorrectly (closed, wrong engine, etc.)."""


class DFSSession:
    """Reuse Featuretools plan + DuckDB RDB tables across train/val/test targets.

    The first ``compute`` freezes the feature list, SQL, and RDB
    ingest. Later ``compute`` calls only replace ``__target__`` / cutoff tables,
    re-execute SQL, and assemble. Column order is fixed after the first success.

    The RDB is snapshotted into DuckDB on the first ``compute``;
    later changes to ``rdb`` are not reflected.
    """

    def __init__(
        self,
        rdb: RDB,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config: Optional[DFSConfig] = None,
    ):
        if config is None:
            config = DFSConfig()
        effective = config.copy(deep=True)

        if effective.engine != "dfs2sql":
            raise DFSSessionError(
                f"DFSSession currently requires engine='dfs2sql' (got {effective.engine!r})."
            )

        # Match compute_dfs_features key handling before any work.
        rdb = rdb.canonicalize_key_types()
        rdb.validate_key_consistency()

        self._rdb = rdb
        self._key_mappings = dict(key_mappings)
        self._cutoff_time_column = cutoff_time_column
        self._config = effective
        self._engine: DFS2SQLEngine = get_dfs_engine("dfs2sql", effective)  # type: ignore[assignment]

        self._builder: Optional[DuckDBBuilder] = None
        # Set only when the session generated the DB path itself; close() deletes it.
        self._temp_db_path: Optional[str] = None
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
        self.n_target_ingest_calls = 0

    def __enter__(self) -> "DFSSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the DuckDB connection.

        Deletes the DB file only if the session generated a temporary path
        (``config.engine_path`` is ``None``); a user-supplied path is kept.
        """
        if self._closed:
            return
        self._closed = True
        if self._builder is not None:
            try:
                self._builder.db.close()
            except Exception as exc:
                logger.debug("DFSSession: error closing DuckDB: {}", exc)
            self._builder = None
        path = self._temp_db_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError as exc:
                logger.debug("DFSSession: could not remove temp db file {}: {}", path, exc)
        self._temp_db_path = None

    def compute(self, target_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute DFS features for one target frame, reusing warm state when available.

        After the first compute, an empty target still returns the frozen
        feature columns (with zero rows), so every split has the same columns.
        """
        self._ensure_open()
        # Planning needs rows; before the first plan, match one-shot and return as-is.
        if len(target_dataframe) == 0 and not self._prepared:
            return target_dataframe

        target_df_with_index, target_df_for_engine = self._prepare_target_frame(
            target_dataframe
        )

        just_planned = not self._prepared
        if just_planned:
            self._plan_and_ingest_rdb(target_df_for_engine)
        if not self._features:
            # Match one-shot: no features means the target comes back unchanged.
            return target_df_with_index.drop(columns="__target_index__")
        if not just_planned:
            self._ingest_target(target_df_for_engine)

        assert self._builder is not None
        assert self._sqls is not None
        assert self._sql_column_order is not None

        feature_matrix = self._engine.execute_and_assemble(
            self._sqls,
            db=self._builder.db,
            target_dataframe=target_df_for_engine,
            column_order=self._sql_column_order,
            config=self._config,
            cutoff_time_col_name=self._cutoff_time_col_name,
        )

        if self._feature_renames:
            feature_matrix = feature_matrix.rename(columns=self._feature_renames, copy=False)

        if self.feature_columns_ is None:
            # Stable output feature column order (after renames), excluding target index.
            self.feature_columns_ = [
                c for c in feature_matrix.columns if c != "__target_index__"
            ]
            self.n_features_ = len(self.feature_columns_)
        else:
            feature_matrix = feature_matrix[["__target_index__"] + self.feature_columns_]

        return self._engine._join_features_to_target(
            target_df_with_index, feature_matrix, target_dataframe.index
        )

    # ------------------------------------------------------------------ internals

    def _ensure_open(self) -> None:
        if self._closed:
            raise DFSSessionError("DFSSession is closed.")

    def _prepare_target_frame(
        self, target_dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Copy so key coercion does not mutate the caller's frame across splits.
        frame = target_dataframe.copy()
        coerce_target_key_types(self._rdb, frame, self._key_mappings)
        target_df_with_index, target_df_for_engine = self._engine._prepare_target_for_engine(
            frame, self._key_mappings, self._cutoff_time_column
        )
        return target_df_with_index, self._apply_pinned_dtypes(target_df_for_engine)

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

    def _ingest_target(self, target_df_for_engine: pd.DataFrame) -> None:
        assert self._builder is not None
        self._engine.ingest_or_replace_target(
            self._builder,
            target_df_for_engine,
            "__target_index__",
            self._cutoff_time_column,
            replace=True,
        )
        self.n_target_ingest_calls += 1

    def _plan_and_ingest_rdb(self, target_df_for_engine: pd.DataFrame) -> None:
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
            "DFSSession: planned {} features (plan call #{})",
            len(features),
            self.n_plan_calls,
        )
        if not features:
            logger.warning("No features generated, check your configuration or data.")
            self.feature_columns_ = []
            self.n_features_ = 0
            self._prepared = True
            return

        self._feature_renames = dfs_feature_renames(features)

        builder, engine_path = self._engine.open_builder(self._config)
        self._builder = builder
        if self._config.engine_path is None:
            self._temp_db_path = engine_path
        self._engine.ingest_rdb_tables(builder, self._rdb)
        self._ingest_target(target_df_for_engine)

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
