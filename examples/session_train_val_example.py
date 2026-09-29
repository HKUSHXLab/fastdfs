"""
Train/val (or test) feature generation with DFSSession.

Demonstrates plan-once / ingest-once reuse when the RDB is shared across splits.
Requires engine=\"dfs2sql\".
"""

from loguru import logger

logger.enable("fastdfs")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastdfs
from fastdfs.transform import (
    CanonicalizeTypes,
    FeaturizeDatetime,
    FillMissingPrimaryKey,
    FilterColumn,
    HandleDummyTable,
    RDBTransformPipeline,
    RDBTransformWrapper,
)
from fastdfs.utils.logging_config import configure_logging

configure_logging(level="INFO")


def main():
    print("=== FastDFS DFSSession train/val example ===")

    rdb_path = project_root / "tests" / "data" / "test_rdb_new"
    rdb = fastdfs.load_rdb(str(rdb_path))

    transform_pipeline = RDBTransformPipeline(
        [
            HandleDummyTable(),
            FillMissingPrimaryKey(),
            RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
            RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
            RDBTransformWrapper(CanonicalizeTypes()),
        ]
    )
    transformed_rdb = transform_pipeline(rdb)

    user_data = np.load(rdb_path / "data" / "user.npz", allow_pickle=True)
    item_data = np.load(rdb_path / "data" / "item.npz", allow_pickle=True)
    user_ids = user_data["user_id"][:3]
    item_ids = item_data["item_id"][:3]

    train_df = pd.DataFrame(
        {
            "user_id": [user_ids[0], user_ids[1], user_ids[2]],
            "item_id": [item_ids[0], item_ids[1], item_ids[2]],
            "timestamp": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-02 11:00:00", "2023-01-03 12:00:00"]
            ),
        }
    )
    val_df = pd.DataFrame(
        {
            "user_id": [user_ids[0], user_ids[1]],
            "item_id": [item_ids[1], item_ids[0]],
            "timestamp": pd.to_datetime(["2023-01-04 13:00:00", "2023-01-05 14:00:00"]),
        }
    )

    key_mappings = {"user_id": "user.user_id", "item_id": "item.item_id"}
    config = fastdfs.DFSConfig(
        engine="dfs2sql",
        engine_path=":memory:",
        max_depth=2,
        agg_primitives=["count", "mean", "max", "min"],
        schema_only_entityset=True,
    )

    with fastdfs.create_dfs_session(
        transformed_rdb,
        key_mappings,
        cutoff_time_column="timestamp",
        config=config,
    ) as session:
        train_features = session.compute(train_df)
        val_features = session.compute(val_df)
        print(
            f"plan={session.n_plan_calls} rdb_ingest={session.n_rdb_ingest_calls} "
            f"target_ingest={session.n_target_ingest_calls} sql_exec={session.n_sql_exec_calls}"
        )

    assert list(train_features.columns) == list(val_features.columns)
    print(f"train shape={train_features.shape} val shape={val_features.shape}")
    print(f"shared feature columns: {session.n_features_}")
    print("=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
