"""Unit tests for DFSSession (plan/RDB reuse across target splits)."""

from __future__ import annotations

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_dfs_session, create_rdb
from fastdfs.dfs import DFSConfig
from fastdfs.dfs.session import DFSSessionError


@pytest.fixture
def tiny_rdb():
    users = pd.DataFrame({"user_id": [1, 2, 3], "age": [25.0, 30.0, 35.0]})
    events = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "user_id": [1, 1, 2, 3],
            "amount": [5.0, 7.0, 9.0, 11.0],
            "timestamp": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
        }
    )
    return create_rdb(
        tables={"user": users, "event": events},
        primary_keys={"user": "user_id", "event": "event_id"},
        foreign_keys=[("event", "user_id", "user", "user_id")],
        time_columns={"event": "timestamp"},
    )


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
    with create_dfs_session(
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
    with create_dfs_session(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        train_out = session.compute(train_target.copy())
        val_out = session.compute(val_target.copy())

        assert session.n_plan_calls == 1
        assert session.n_rdb_ingest_calls == 1
        assert session.n_target_ingest_calls == 2
        assert session.n_sql_exec_calls == 2
        assert list(train_out.columns) == list(val_out.columns)
        assert session.feature_columns_ is not None
        assert all(c in train_out.columns for c in session.feature_columns_)


def test_session_close_blocks_compute(tiny_rdb, dfs_config, train_target):
    session = create_dfs_session(
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
        create_dfs_session(
            tiny_rdb,
            {"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=DFSConfig(engine="featuretools"),
        )


def test_session_prepare_then_compute(
    tiny_rdb, dfs_config, train_target, val_target
):
    with create_dfs_session(
        tiny_rdb,
        {"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=dfs_config,
    ) as session:
        session.prepare(train_target.copy())
        assert session.n_plan_calls == 1
        assert session.n_rdb_ingest_calls == 1
        assert session.n_target_ingest_calls == 0

        train_out = session.compute(train_target.copy())
        val_out = session.compute(val_target.copy())
        assert session.n_plan_calls == 1
        assert session.n_rdb_ingest_calls == 1
        assert session.n_target_ingest_calls == 2
        assert list(train_out.columns) == list(val_out.columns)


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
    with create_dfs_session(
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
