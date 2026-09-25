"""Parallel dfs2sql SQL execution parity tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest
from sqlglot import parse_one
from sqlglot.expressions import Select

from fastdfs.api import compute_dfs_features
from fastdfs.dataset.rdb import RDB
from fastdfs.dfs import DFSConfig
from fastdfs.dfs.dfs2sql_engine import DFS2SQLEngine
from fastdfs.dfs.gen_sqls import features2sql
from fastdfs.dfs.merge_sql import merge_queries


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
            "user_id": [
                user_ids[0],
                user_ids[1],
                user_ids[2],
                user_ids[0],
                user_ids[1],
            ],
            "item_id": [
                item_ids[0],
                item_ids[1],
                item_ids[2],
                item_ids[1],
                item_ids[0],
            ],
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
def test_merge_flag_controls_all_grouping(
    rdb_dataset: RDB,
    target_dataframe: pd.DataFrame,
    key_mappings: Dict[str, str],
    use_cutoff: bool,
) -> None:
    generated: List[int] = []

    def generate(*args: Any, **kwargs: Any) -> List[Select]:
        queries = features2sql(*args, **kwargs)
        assert len(queries) == len(args[0])
        generated.append(len(queries))
        return queries

    results = []
    executed = []
    original_execute = DFS2SQLEngine._execute_feature_sqls

    def execute(
        engine: DFS2SQLEngine, sqls: List[Select], **kwargs: Any
    ) -> List[pd.DataFrame]:
        executed.append(len(sqls))
        return original_execute(engine, sqls, **kwargs)

    for enabled in (False, True):
        with (
            patch("fastdfs.dfs.dfs2sql_engine.features2sql", side_effect=generate),
            patch(
                "fastdfs.dfs.dfs2sql_engine.merge_queries", wraps=merge_queries
            ) as merger,
            patch.object(DFS2SQLEngine, "_execute_feature_sqls", execute),
        ):
            results.append(
                compute_dfs_features(
                    rdb=rdb_dataset,
                    target_dataframe=target_dataframe,
                    key_mappings=key_mappings,
                    cutoff_time_column="interaction_time" if use_cutoff else None,
                    config_overrides={
                        "engine": "dfs2sql",
                        "engine_path": ":memory:",
                        "max_depth": 3,
                        "agg_primitives": ["min", "max", "mean"],
                        "use_cutoff_time": use_cutoff,
                        "dfs2sql_merge_queries": enabled,
                        "dfs2sql_threads": 2,
                    },
                )
            )
            assert merger.call_count == int(enabled)

    assert generated[0] == generated[1]
    assert executed[0] == generated[0]
    assert executed[1] < executed[0]
    pd.testing.assert_frame_equal(results[0], results[1], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("use_cutoff", [False, True])
@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("include_cutoff", [False, True])
def test_parallel_matches_sequential(
    rdb_dataset,
    target_dataframe,
    key_mappings,
    use_cutoff,
    workers,
    in_memory,
    depth,
    include_cutoff,
    tmp_path,
):
    repeated = target_dataframe.iloc[[0]].copy()
    repeated["interaction_time"] = pd.Timestamp("2024-01-03")
    target_dataframe = pd.concat([target_dataframe, repeated], ignore_index=True)
    overrides = {
        "engine": "dfs2sql",
        "max_depth": depth,
        "agg_primitives": ["count", "mean", "min", "max", "std"],
        "use_cutoff_time": use_cutoff,
        "include_cutoff_time": include_cutoff,
    }
    cutoff = "interaction_time" if use_cutoff else None
    seq = compute_dfs_features(
        rdb=rdb_dataset,
        target_dataframe=target_dataframe,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff,
        config_overrides={
            **overrides,
            "dfs2sql_sql_workers": 1,
            "dfs2sql_merge_queries": False,
            "engine_path": str(tmp_path / "reference.db"),
        },
    )
    par = compute_dfs_features(
        rdb=rdb_dataset,
        target_dataframe=target_dataframe,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff,
        config_overrides={
            **overrides,
            "dfs2sql_sql_workers": workers,
            "engine_path": ":memory:" if in_memory else str(tmp_path / "candidate.db"),
            "dfs2sql_threads": 2,
            "dfs2sql_memory_limit": "256MB",
            "dfs2sql_temp_directory": str(tmp_path / "spill"),
        },
    )
    pd.testing.assert_frame_equal(seq, par, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_workers_preserve_threads_and_share_memory(workers: int) -> None:
    config = DFSConfig(dfs2sql_sql_workers=workers)
    engine = DFS2SQLEngine(config)
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=3")
        db.execute("CREATE TABLE source AS SELECT 7 AS value")
        frames = engine._execute_feature_sqls(
            [parse_one("SELECT value FROM source") for _ in range(4)],
            db=db,
            config=config,
            cutoff_time_col_name=None,
        )
        assert [frame.iloc[0, 0] for frame in frames] == [7] * 4
        assert db.execute("SELECT current_setting('threads')").fetchone()[0] == 3


def test_worker_error_leaves_database_usable() -> None:
    config = DFSConfig(dfs2sql_sql_workers=2)
    with duckdb.connect(":memory:") as db:
        with pytest.raises(duckdb.CatalogException):
            DFS2SQLEngine(config)._execute_feature_sqls(
                [parse_one("SELECT * FROM missing"), parse_one("SELECT 1")],
                db=db,
                config=config,
                cutoff_time_col_name=None,
            )
        assert db.execute("SELECT 1").fetchone() == (1,)
