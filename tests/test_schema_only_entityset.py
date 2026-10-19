"""Tests for schema-only Featuretools EntitySet planning."""

from __future__ import annotations

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features
from fastdfs.dfs import DFSConfig, get_dfs_engine


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
