"""Tests for schema-only Featuretools EntitySet planning."""

from __future__ import annotations

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig


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
def target():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-05", "2020-01-05"]),
        }
    )


def test_schema_only_matches_full_entityset_features(tiny_rdb, target):
    base = dict(
        max_depth=2,
        engine="dfs2sql",
        engine_path=":memory:",
        agg_primitives=["count", "sum", "mean"],
        dfs2sql_sql_workers=1,
        dfs2sql_merge_queries=True,
    )
    full = compute_dfs_features(
        tiny_rdb,
        target.copy(),
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=DFSConfig(**base, schema_only_entityset=False),
    )
    schema = compute_dfs_features(
        tiny_rdb,
        target.copy(),
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="timestamp",
        config=DFSConfig(**base, schema_only_entityset=True),
    )
    assert list(full.columns) == list(schema.columns)
    pd.testing.assert_frame_equal(
        full.reset_index(drop=True),
        schema.reset_index(drop=True),
        check_dtype=False,
    )
