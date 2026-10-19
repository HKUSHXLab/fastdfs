"""Tests for schema-only Featuretools EntitySet planning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig, DFSSession, get_dfs_engine


@pytest.fixture
def target():
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-05", "2020-01-05"]),
        }
    )


@pytest.fixture
def typed_rdb():
    """RDB whose raw dtypes differ from what full planning converts them to."""
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "age": [25, 30, 35],  # whole numbers in a float column
            "score": np.array([0.5, 1.5, 2.5], dtype="float32"),
            "country": ["us", "cn", "us"],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "user_id": [1, 1, 2, 3],
            "amount": [5.0, 7.0, 9.0, 11.0],
            "kind": ["a", "b", "b", "a"],
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
        type_hints={
            "user": {"age": "float", "score": "float", "country": "category"},
            "event": {"kind": "category"},
        },
    )


def _typed_config(schema_only_entityset):
    return DFSConfig(
        max_depth=2,
        engine="dfs2sql",
        engine_path=":memory:",
        agg_primitives=["count", "sum", "mean", "mode"],
        schema_only_entityset=schema_only_entityset,
    )


def test_schema_only_matches_full_entityset_features(typed_rdb, target):
    outputs = [
        compute_dfs_features(
            typed_rdb,
            target.copy(),
            key_mappings={"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=_typed_config(schema_only),
        )
        for schema_only in (False, True)
    ]
    full, schema = outputs
    assert full["d1__user.country"].dtype == "category"
    assert full["d1__user.score"].dtype == "float64"
    pd.testing.assert_frame_equal(full, schema)


def test_session_schema_only_matches_full_entityset_features(typed_rdb, target):
    outputs = []
    for schema_only in (False, True):
        with DFSSession(
            typed_rdb,
            {"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=_typed_config(schema_only),
        ) as session:
            outputs.append(session.compute(target.copy()))
    pd.testing.assert_frame_equal(*outputs)


def test_schema_only_rejected_for_featuretools_engine(tiny_rdb, target):
    with pytest.raises(ValueError, match="schema_only_entityset=True requires engine='dfs2sql'"):
        compute_dfs_features(
            tiny_rdb,
            target.copy(),
            key_mappings={"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config=DFSConfig(engine="featuretools", schema_only_entityset=True),
        )


def test_schema_only_rejected_via_config_overrides(tiny_rdb, target):
    engine = get_dfs_engine("featuretools", DFSConfig(engine="featuretools"))
    with pytest.raises(ValueError, match="schema_only_entityset=True requires engine='dfs2sql'"):
        engine.compute_features(
            tiny_rdb.canonicalize_key_types(),
            target.assign(user_id=target["user_id"].astype(str)),
            key_mappings={"user_id": "user.user_id"},
            cutoff_time_column="timestamp",
            config_overrides={"schema_only_entityset": True},
        )
