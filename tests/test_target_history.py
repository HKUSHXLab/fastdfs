import pandas as pd
import pytest
from fastdfs.api import compute_dfs_features, create_rdb
from fastdfs.dfs import DFSConfig
from loguru import logger
import numpy as np

def test_target_history_flow():
    """Integration test for adding target history and generating features."""

    # 1. Create RDB
    user_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 35]
    })

    # Use create_rdb helper for convenience
    rdb = create_rdb(
        tables={"user": user_df},
        primary_keys={"user": "user_id"}
    )

    # 2. Create Target DF (Data to be used as history)
    # User 1: 10 at 01-01, 20 at 01-02
    # User 2: 30 at 01-03
    target_df = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        "amount": [10.0, 20.0, 30.0, 40.0]
    })

    # 3. Augment RDB
    key_mappings = {"user_id": "user.user_id"}
    cutoff_time_column = "timestamp"

    logger.info("Augmenting RDB with target history...")
    rdb_augmented = rdb.add_table(
        target_df,
        name="__target_history__",
        time_column=cutoff_time_column,
        foreign_keys=[("user_id", "user", "user_id")]
    )

    assert "__target_history__" in rdb_augmented.table_names
    rels = rdb_augmented.get_relationships()
    assert ("__target_history__", "user_id", "user", "user_id") in rels

    # 4. Run DFS
    # We create a prediction dataframe.
    # User 1 at 2020-01-05 should see history (10, 20) -> Mean 15
    # User 2 at 2020-01-05 should see history (30) -> Mean 30
    # User 3 at 2020-01-01 should see history () -> Mean NaN (since history is at 2020-01-04 which is future)
    # Wait, strict cutoff means purely before.

    prediction_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "timestamp": pd.to_datetime(["2020-01-05", "2020-01-05", "2020-01-01"])
    })

    # Use DFS2SQL engine (default)
    config = DFSConfig(max_depth=2)

    logger.info("Computing DFS features...")
    features = compute_dfs_features(
        rdb=rdb_augmented,
        target_dataframe=prediction_df,
        key_mappings=key_mappings,
        cutoff_time_column="timestamp",
        config=config,
        config_overrides={"engine": "dfs2sql"}
    )

    cols = features.columns.tolist()
    logger.info(f"Generated features: {cols}")
    logger.info(f"Features head:\n{features.head()}")
    logger.info(f"Features dtypes:\n{features.dtypes}")

    # 5. Verify features
    # Expect aggregated features from __target_history__.amount
    # Naming might be: user.MEAN(__target_history__.amount) or similar

    mean_cols = [c for c in cols if "amount" in c and "mean" in c.lower()]
    assert len(mean_cols) > 0, f"No mean feature for amount found in {cols}"

    mean_col = mean_cols[0]
    logger.info(f"Checking features from column: {mean_col}")

    # Check User 1: (10+20)/2 = 15
    # Use iloc to avoid potential type mismatch on user_id (int vs str)
    u1_val = features.iloc[0][mean_col]
    assert abs(u1_val - 15.0) < 0.001, f"User 1 expected 15.0, got {u1_val}"

    # Check User 2: 30
    u2_val = features.iloc[1][mean_col]
    assert abs(u2_val - 30.0) < 0.001, f"User 2 expected 30.0, got {u2_val}"

    # Check User 3: NaN (timestamp 2020-01-01 is before history 2020-01-04)
    u3_val = features.iloc[2][mean_col]
    assert pd.isna(u3_val), f"User 3 expected NaN, got {u3_val}"