
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastdfs.api import compute_dfs_features
from fastdfs.dataset.rdb import RDBDataset
from fastdfs.dfs import DFSConfig

@pytest.fixture
def test_data_path():
    """Path to new RDB test dataset."""
    return Path(__file__).parent / "data" / "test_rdb_new"

@pytest.fixture
def rdb_dataset(test_data_path):
    """Load RDB dataset directly from new test data."""
    return RDBDataset(test_data_path)

@pytest.fixture
def complex_target_dataframe(rdb_dataset):
    """Create a target dataframe with complex column types."""
    # Get some real user and item IDs from the dataset
    user_table = rdb_dataset.get_table('user')
    item_table = rdb_dataset.get_table('item')

    # Take first few users and items
    user_ids = user_table['user_id'].head(3).tolist()
    item_ids = item_table['item_id'].head(3).tolist()

    df = pd.DataFrame({
        "user_id": [user_ids[0], user_ids[1], user_ids[2]],
        "item_id": [item_ids[0], item_ids[1], item_ids[2]],
        "interaction_time": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03"
        ]),
        # Complex columns that should be ignored by DFS but preserved in output
        "list_col": [[1, 2], [3, 4], [5, 6]],
        "dict_col": [{"a": 1}, {"b": 2}, {"c": 3}],
        "array_col": [np.array([1.1, 2.2]), np.array([3.3, 4.4]), np.array([5.5, 6.6])],
        "mixed_col": [1, "string", [1, 2]]
    })
    return df

@pytest.mark.parametrize("engine_name", ["featuretools", "dfs2sql"])
def test_complex_target_columns(rdb_dataset, complex_target_dataframe, engine_name):
    """
    Test that target dataframes with complex column types (lists, dicts, arrays)
    are handled correctly.
    
    The DFS engine should:
    1. Ignore these columns during feature generation (thanks to our optimization)
    2. Preserve them in the final output
    """
    
    # Configure DFS
    config = DFSConfig(
        engine=engine_name,
        max_depth=1,
        agg_primitives=["count"],
        engine_path="/tmp/test_complex_cols.db"  # Only used for dfs2sql
    )
    
    # Run DFS
    features = compute_dfs_features(
        rdb=rdb_dataset,
        target_dataframe=complex_target_dataframe,
        key_mappings={
            "user_id": "user.user_id",
            "item_id": "item.item_id"
        },
        cutoff_time_column="interaction_time",
        config=config
    )
    
    # Assertions
    
    # 1. Check that execution succeeded (implied by reaching here)
    assert features is not None
    
    # 2. Check that original complex columns are preserved
    assert "list_col" in features.columns
    assert "dict_col" in features.columns
    assert "array_col" in features.columns
    assert "mixed_col" in features.columns
    
    # 3. Verify content of complex columns matches input
    pd.testing.assert_series_equal(
        features["list_col"], 
        complex_target_dataframe["list_col"],
        check_names=False
    )
    
    # 4. Check that new features were actually generated
    # We expect at least some features from depth=1 count
    new_features = [c for c in features.columns if c not in complex_target_dataframe.columns]
    assert len(new_features) > 0
    
    # 5. Verify row count matches
    assert len(features) == len(complex_target_dataframe)

