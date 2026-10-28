"""Unit tests for DFSSession (plan/RDB reuse across target splits)."""

from __future__ import annotations

import gc
import os
import tempfile

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features
from fastdfs.dfs import DFSConfig
from fastdfs.dfs.dfs2sql_engine import DFS2SQLEngine
from fastdfs.dfs.session import DFSSession, DFSSessionError


@pytest.fixture
def dfs_config():
    return DFSConfig(
        max_depth=2,
        engine="dfs2sql",
        engine_path=":memory:",
        agg_primitives=["count", "sum", "mean"],
        dfs2sql_sql_workers=1,
        dfs2sql_merge_queries=True,
    )


@pytest.fixture
def train_target():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-05", "2020-01-05"]),
        }
    )


@pytest.fixture
def val_target():
    return pd.DataFrame(
        {
            "user_id": [2, 3],
            "timestamp": pd.to_datetime(["2020-01-06", "2020-01-06"]),
        }
    )


def test_session_parity_with_oneshot(tiny_rdb, dfs_config, train_target):
    oneshot = compute_dfs_features(
        tiny_rdb,
        train_target.copy(),
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    )
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        warmed = session.compute(train_target.copy())

    assert list(warmed.columns) == list(oneshot.columns)
    pd.testing.assert_frame_equal(
        warmed.reset_index(drop=True),
        oneshot.reset_index(drop=True),
        check_dtype=False,
    )


def test_session_reuses_plan_and_rdb_ingest(
    tiny_rdb, dfs_config, train_target, val_target
):
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        train_out = session.compute(train_target.copy())
        val_out = session.compute(val_target.copy())

        assert session._n_plan_calls == 1
        assert session._n_target_ingest_calls == 2
        assert list(train_out.columns) == list(val_out.columns)
        assert session.feature_columns_ is not None
        assert all(c in train_out.columns for c in session.feature_columns_)


def test_session_close_blocks_compute(tiny_rdb, dfs_config, train_target):
    session = DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    )
    session.compute(train_target.copy())
    session.close()
    with pytest.raises(DFSSessionError, match="closed"):
        session.compute(train_target.copy())


def test_session_rejects_non_dfs2sql(tiny_rdb, train_target):
    with pytest.raises(DFSSessionError, match="dfs2sql"):
        DFSSession(
            tiny_rdb,
            {"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=DFSConfig(engine="featuretools"),
        )


def test_session_val_matches_oneshot_val(
    tiny_rdb, dfs_config, train_target, val_target
):
    oneshot_val = compute_dfs_features(
        tiny_rdb,
        val_target.copy(),
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    )
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        session.compute(train_target.copy())
        warmed_val = session.compute(val_target.copy())

    # Same columns as train-planned session; values should match one-shot val
    # on the intersection of feature columns (session freezes train plan).
    shared = [c for c in warmed_val.columns if c in oneshot_val.columns]
    assert shared == list(warmed_val.columns)
    pd.testing.assert_frame_equal(
        warmed_val[shared].reset_index(drop=True),
        oneshot_val[shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_reserved_target_index_column_raises(tiny_rdb, dfs_config, train_target):
    target = train_target.assign(__target_index__=[5, 6])
    with pytest.raises(ValueError, match="reserved column name"):
        compute_dfs_features(
            tiny_rdb,
            target.copy(),
            key_mappings={"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=dfs_config,
        )
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        with pytest.raises(ValueError, match="reserved column name"):
            session.compute(target.copy())


def test_session_zero_features_matches_oneshot(
    tiny_rdb, dfs_config, train_target, val_target
):
    config = dfs_config.model_copy(update={"max_depth": 0})
    oneshot_val = compute_dfs_features(
        tiny_rdb,
        val_target.copy(),
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    )
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    ) as session:
        train_out = session.compute(train_target.copy())
        val_out = session.compute(val_target.copy())
        assert session.feature_columns_ == []
        assert session._n_target_ingest_calls == 0

    assert list(train_out.columns) == list(train_target.columns)
    pd.testing.assert_frame_equal(val_out, oneshot_val)


def test_session_close_keeps_user_engine_path(tmp_path, tiny_rdb, dfs_config, train_target):
    db_path = tmp_path / "my.db"
    config = dfs_config.model_copy(update={"engine_path": str(db_path)})
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    ) as session:
        session.compute(train_target.copy())
    assert db_path.exists()


def test_session_close_removes_temp_engine_path(tiny_rdb, dfs_config, train_target):
    config = dfs_config.model_copy(update={"engine_path": None})
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    ) as session:
        session.compute(train_target.copy())
        temp_path = session._temp_db_path
        assert temp_path is not None and os.path.exists(temp_path)
    assert not os.path.exists(temp_path)


def test_session_empty_split_keeps_feature_columns(tiny_rdb, dfs_config, train_target):
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        train_out = session.compute(train_target.copy())
        empty_out = session.compute(train_target.iloc[:0])

    assert len(empty_out) == 0
    assert all(c in empty_out.columns for c in session.feature_columns_)
    pd.testing.assert_series_equal(empty_out.dtypes, train_out.dtypes)


def test_session_without_close_removes_temp_engine_path(tiny_rdb, dfs_config, train_target):
    config = dfs_config.model_copy(update={"engine_path": None})
    session = DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    )
    session.compute(train_target.copy())
    temp_path = session._temp_db_path
    assert temp_path is not None and os.path.exists(temp_path)

    del session
    gc.collect()
    assert not os.path.exists(temp_path)


def test_session_garbage_collection_after_close_keeps_user_engine_path(
    tmp_path, tiny_rdb, dfs_config, train_target
):
    db_path = tmp_path / "my.db"
    config = dfs_config.model_copy(update={"engine_path": str(db_path)})
    session = DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    )
    session.compute(train_target.copy())
    session.close()
    session.close()

    del session
    gc.collect()
    assert db_path.exists()


@pytest.mark.parametrize("user_engine_path", [True, False])
def test_session_retry_after_failed_first_compute(
    tmp_path, monkeypatch, tiny_rdb, dfs_config, train_target, user_engine_path
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    db_path = tmp_path / "my.db"
    config = dfs_config.model_copy(
        update={"engine_path": str(db_path) if user_engine_path else None}
    )
    original = DFS2SQLEngine.plan_feature_sqls
    calls = {"n": 0}

    def fail_once(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFS2SQLEngine, "plan_feature_sqls", fail_once)
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=config,
    ) as session:
        with pytest.raises(RuntimeError, match="transient failure"):
            session.compute(train_target.copy())
        assert list(tmp_path.glob("fastdfs_*.db*")) == []

        out = session.compute(train_target.copy())
        assert all(c in out.columns for c in session.feature_columns_)

    assert list(tmp_path.glob("fastdfs_*.db*")) == []
    assert db_path.exists() == user_engine_path


def test_session_compute_does_not_modify_target(tiny_rdb, dfs_config, train_target):
    target = train_target.assign(label=[0, 1])
    snapshot = target.copy(deep=True)
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        out = session.compute(target)

    assert out["user_id"].dtype == object  # keys were coerced in the output...
    pd.testing.assert_frame_equal(target, snapshot)  # ...but not in the caller's frame


def test_session_empty_first_split_raises(tiny_rdb, dfs_config, train_target):
    with DFSSession(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        with pytest.raises(DFSSessionError, match="needs at least one row"):
            session.compute(train_target.iloc[:0])
        assert session.feature_columns_ is None

        # The session is still usable: plan on a non-empty split, then empty is fine.
        train_out = session.compute(train_target.copy())
        empty_out = session.compute(train_target.iloc[:0])
        assert list(empty_out.columns) == list(train_out.columns)
