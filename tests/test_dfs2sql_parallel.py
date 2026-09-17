"""Parallel dfs2sql SQL execution parity tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fastdfs.api import compute_dfs_features
from fastdfs.dataset.rdb import RDB
from fastdfs.dfs import DFSConfig


@pytest.fixture
def test_data_path():
    return Path(__file__).parent / "data" / "test_rdb_new"


@pytest.fixture
def rdb_dataset(test_data_path):
    return RDB(test_data_path)


@pytest.fixture
def target_dataframe(rdb_dataset):
    user_table = rdb_dataset.get_table("user")
    item_table = rdb_dataset.get_table("item")
    user_ids = user_table["user_id"].head(3).tolist()
    item_ids = item_table["item_id"].head(3).tolist()
    return pd.DataFrame(
        {
            "user_id": [user_ids[0], user_ids[1], user_ids[2], user_ids[0], user_ids[1]],
            "item_id": [item_ids[0], item_ids[1], item_ids[2], item_ids[1], item_ids[0]],
            "interaction_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
        }
    )


@pytest.fixture
def key_mappings():
    return {"user_id": "user.user_id", "item_id": "item.item_id"}


def test_dfs2sql_sql_workers_default():
    assert DFSConfig().dfs2sql_sql_workers == 1


@pytest.mark.parametrize("use_cutoff", [False, True])
def test_parallel_matches_sequential(
    rdb_dataset, target_dataframe, key_mappings, use_cutoff
):
    overrides = {
        "engine": "dfs2sql",
        "max_depth": 2,
        "agg_primitives": ["count", "mean", "min", "max"],
        "use_cutoff_time": use_cutoff,
    }
    cutoff = "interaction_time" if use_cutoff else None
    seq = compute_dfs_features(
        rdb=rdb_dataset,
        target_dataframe=target_dataframe,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff,
        config_overrides={**overrides, "dfs2sql_sql_workers": 1},
    )
    par = compute_dfs_features(
        rdb=rdb_dataset,
        target_dataframe=target_dataframe,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff,
        config_overrides={**overrides, "dfs2sql_sql_workers": 4},
    )
    assert set(seq.columns) == set(par.columns)
    assert len(seq) == len(par)
    for col in seq.columns:
        if pd.api.types.is_numeric_dtype(seq[col]):
            pd.testing.assert_series_equal(
                seq[col].reset_index(drop=True),
                par[col].reset_index(drop=True),
                check_names=False,
                rtol=1e-5,
                atol=1e-6,
            )
