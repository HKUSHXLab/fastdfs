"""Warm DFS session: plan + RDB ingest once, recompute per target split."""

from __future__ import annotations

import weakref
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
from .dfs2sql_engine import DFS2SQLEngine, release_duckdb
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
        effective = config.model_copy(deep=True)

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
        # Runs release_duckdb on close(), garbage collection or interpreter exit, at most once.
        self._finalizer: Optional[weakref.finalize] = None
        self._features: Optional[List[Any]] = None
        self._sqls: Optional[List[Any]] = None
        self._sql_column_order: Optional[List[str]] = None
        self._cutoff_time_col_name: Optional[str] = None
        self._feature_renames: Dict[str, str] = {}
        # The first split's cutoff dtype. Later splits must have the same kind
        # (numeric or datetime), because the frozen SQL compares cutoffs with the
        # RDB's time columns; empty later splits are cast to it.
        self._cutoff_dtype: Optional[Any] = None
        self.feature_columns_: Optional[List[str]] = None
        self.n_features_: int = 0

        self._prepared = False
        self._closed = False

        # Counters for tests / diagnostics
        self._n_plan_calls = 0
        self._n_target_ingest_calls = 0

    def __enter__(self) -> "DFSSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the DuckDB connection.

        Deletes the DB file only if the session generated a temporary path
        (``config.engine_path`` is ``None``); a user-supplied path is not deleted
        on close. Note that the first ``compute()`` replaces any existing file at a
        user-supplied ``engine_path``, so don't point it at a database you want to
        keep, or share it with another session or call that is still open.
        The same cleanup also runs if the session is garbage-collected or the
        interpreter exits without ``close()``.
        """
        if self._closed:
            return
        self._closed = True
        if self._finalizer is not None:
            self._finalizer()
        self._builder = None

    def compute(self, target_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute DFS features for one target frame, reusing warm state when available.

        After the first compute, an empty target still returns the frozen
        feature columns (with zero rows), so every split has the same columns.
        The first compute must have at least one row, because it plans the
        features; an empty first target raises ``DFSSessionError``.
        """
        self._ensure_open()
        # Planning needs rows. Returning the frame unchanged here would silently give
        # this split fewer columns than every later one, so fail loudly instead.
        if len(target_dataframe) == 0 and not self._prepared:
            raise DFSSessionError(
                "The first DFSSession.compute() needs at least one row to plan features; "
                "compute a non-empty split (e.g. train) first. Later empty splits are fine."
            )

        target_df_with_index, target_df_for_engine = self._prepare_target_frame(
            target_dataframe
        )

        cutoff_col = self._cutoff_time_column
        if self._prepared and cutoff_col is not None:
            if len(target_df_for_engine) == 0:
                # An empty frame's dtype is arbitrary (e.g. float64 for pd.DataFrame
                # with no rows); there are no values to misread, so cast it.
                target_df_for_engine[cutoff_col] = target_df_for_engine[cutoff_col].astype(
                    self._cutoff_dtype
                )
            else:
                is_numeric = pd.api.types.is_numeric_dtype
                this_kind, first_kind = (
                    "numeric" if is_numeric(dtype) else "datetime"
                    for dtype in (target_df_for_engine[cutoff_col].dtype, self._cutoff_dtype)
                )
                if this_kind != first_kind:
                    raise DFSSessionError(
                        f"Cutoff time column '{cutoff_col}' is {this_kind} in this split but was "
                        f"{first_kind} in the first split; all splits must use the same kind."
                    )

        just_planned = not self._prepared
        if just_planned:
            self._plan_and_ingest_rdb(target_df_for_engine)
            if cutoff_col is not None:
                self._cutoff_dtype = target_df_for_engine[cutoff_col].dtype
        if not self._features:
            # Match one-shot: no features means the target comes back unchanged.
            return target_df_with_index.drop(columns="__target_index__")
        if not just_planned:
            assert self._builder is not None
            self._ingest_target(self._builder, target_df_for_engine)

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
        # Shallow copy: key coercion replaces columns in the copy only, so the caller's
        # frame is untouched without duplicating all of its (possibly wide) data.
        frame = target_dataframe.copy(deep=False)
        coerce_target_key_types(self._rdb, frame, self._key_mappings)
        target_df_with_index, target_df_for_engine = self._engine._prepare_target_for_engine(
            frame, self._key_mappings, self._cutoff_time_column
        )
        return target_df_with_index, target_df_for_engine

    def _ingest_target(self, builder: DuckDBBuilder, target_df_for_engine: pd.DataFrame) -> None:
        self._engine.ingest_or_replace_target(
            builder,
            target_df_for_engine,
            "__target_index__",
            self._cutoff_time_column,
            list(self._key_mappings),
            replace=True,
        )
        self._n_target_ingest_calls += 1

    def _plan_and_ingest_rdb(self, target_df_for_engine: pd.DataFrame) -> None:
        features = self._engine.prepare_features(
            self._rdb,
            target_df_for_engine,
            self._key_mappings,
            self._cutoff_time_column,
            self._config,
        )
        self._n_plan_calls += 1
        self._features = features
        logger.info(
            "DFSSession: planned {} features (plan call #{})",
            len(features),
            self._n_plan_calls,
        )
        if not features:
            logger.warning("No features generated, check your configuration or data.")
            self.feature_columns_ = []
            self.n_features_ = 0
            self._prepared = True
            return

        self._feature_renames = dfs_feature_renames(features)

        builder, engine_path = self._engine.open_builder(self._config)
        temp_db_path = engine_path if self._config.engine_path is None else None
        finalizer = weakref.finalize(self, release_duckdb, builder.db, temp_db_path)
        try:
            self._engine.ingest_rdb_tables(builder, self._rdb, features)
            self._ingest_target(builder, target_df_for_engine)
            sqls, column_order, cutoff_time_col_name = self._engine.plan_feature_sqls(
                builder,
                features,
                self._cutoff_time_column,
                self._config,
            )
        except BaseException:
            # Leave the session unprepared with nothing open, so a retry starts clean.
            finalizer()
            raise

        self._builder = builder
        self._finalizer = finalizer
        self._sqls = sqls
        self._sql_column_order = column_order
        self._cutoff_time_col_name = cutoff_time_col_name
        self._prepared = True
