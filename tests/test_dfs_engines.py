"""
Tests for the new DFS engine interface (Phase 2).

This test file validates that the new DFS engines correctly compute features
for external target dataframes using the simplified RDB dataset interface.
"""

import pytest
import pandas as pd
import tempfile
import numpy as np
from pathlib import Path

from fastdfs.dfs import DFSConfig, get_dfs_engine, FeaturetoolsEngine, DFS2SQLEngine
from fastdfs.dataset.rdb_simplified import RDBDataset, convert_task_dataset_to_rdb
from fastdfs.api_new import load_rdb, compute_dfs_features, DFSPipeline


@pytest.fixture
def test_data_path():
    """Path to existing test dataset."""
    return Path(__file__).parent / "data" / "test_rdb"

@pytest.fixture 
def rdb_dataset(test_data_path):
    """Create RDB dataset from existing test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        rdb_path = Path(temp_dir) / "rdb"
        
        # Convert task-based dataset to RDB-only format
        convert_task_dataset_to_rdb(test_data_path, rdb_path)
        
        # Load the converted RDB dataset
        yield RDBDataset(rdb_path)

@pytest.fixture
def target_dataframe(rdb_dataset):
    """Create a target dataframe using actual IDs from the dataset."""
    # Get some real user and item IDs from the dataset
    user_table = rdb_dataset.get_table('user')
    item_table = rdb_dataset.get_table('item')
    
    # Take first few users and items
    user_ids = user_table['user_id'].head(3).tolist()
    item_ids = item_table['item_id'].head(3).tolist()
    
    return pd.DataFrame({
        "user_id": [user_ids[0], user_ids[1], user_ids[2], user_ids[0], user_ids[1]],
        "item_id": [item_ids[0], item_ids[1], item_ids[2], item_ids[1], item_ids[0]],
        "interaction_time": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", 
            "2024-01-04", "2024-01-05"
        ])
    })

@pytest.fixture
def key_mappings():
    """Key mappings for test target dataframe."""
    return {
        "user_id": "user.user_id",
        "item_id": "item.item_id"
    }


class TestDFSConfig:
    """Test DFS configuration."""
    
    def test_default_config(self):
        """Test default DFS configuration."""
        config = DFSConfig()
        
        assert config.max_depth == 2
        assert config.use_cutoff_time == True
        assert config.engine == "featuretools"
        assert "count" in config.agg_primitives
        assert "mean" in config.agg_primitives
    
    def test_custom_config(self):
        """Test custom DFS configuration."""
        config = DFSConfig(
            max_depth=3,
            engine="dfs2sql",
            agg_primitives=["count", "sum"],
            use_cutoff_time=False
        )
        
        assert config.max_depth == 3
        assert config.engine == "dfs2sql"
        assert config.agg_primitives == ["count", "sum"]
        assert config.use_cutoff_time == False


class TestEngineRegistration:
    """Test DFS engine registration system."""
    
    def test_get_dfs_engine_featuretools(self):
        """Test getting the Featuretools DFS engine."""
        config = DFSConfig(engine="featuretools")
        engine = get_dfs_engine("featuretools", config)
        assert isinstance(engine, FeaturetoolsEngine)
    
    def test_get_dfs_engine_dfs2sql(self):
        """Test getting the DFS2SQL engine."""
        config = DFSConfig(engine="dfs2sql")
        engine = get_dfs_engine("dfs2sql", config)
        assert isinstance(engine, DFS2SQLEngine)


class TestFeaturetoolsEngine:
    """Test Featuretools engine implementation."""
    
    def test_engine_registration(self):
        """Test that Featuretools engine is properly registered."""
        config = DFSConfig(engine="featuretools")
        engine = get_dfs_engine("featuretools", config)
        assert isinstance(engine, FeaturetoolsEngine)
        assert engine.name == "featuretools"
    
    def test_compute_features_basic(self, rdb_dataset, target_dataframe, key_mappings):
        """Test basic feature computation with Featuretools engine."""
        config = DFSConfig(
            engine="featuretools",
            max_depth=1,
            agg_primitives=["count", "mean"],
            use_cutoff_time=False
        )
        
        engine = get_dfs_engine("featuretools", config)
        
        # Compute features
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )
        
        # Check that result contains original columns
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
        assert "interaction_time" in result_df.columns
        
        # Check that result has same number of rows
        assert len(result_df) == len(target_dataframe)
        
        # Check that some features were generated
        feature_cols = [col for col in result_df.columns 
                       if col not in target_dataframe.columns]
        assert len(feature_cols) > 0, f"No features generated. Columns: {result_df.columns.tolist()}"
    
    def test_compute_features_with_cutoff_time(self, rdb_dataset, target_dataframe, key_mappings):
        """Test feature computation with cutoff time."""
        config = DFSConfig(
            engine="featuretools",
            max_depth=1,
            agg_primitives=["count"],
            use_cutoff_time=True
        )
        
        engine = get_dfs_engine("featuretools", config)
        
        # Compute features with cutoff time
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column="interaction_time"
        )
        
        # Check that result is valid
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns


class TestDFS2SQLEngine:
    """Test DFS2SQL engine implementation."""
    
    def test_engine_registration(self):
        """Test that DFS2SQL engine is properly registered."""
        config = DFSConfig(engine="dfs2sql")
        engine = get_dfs_engine("dfs2sql", config)
        assert isinstance(engine, DFS2SQLEngine)
        assert engine.name == "dfs2sql"
    
    def test_compute_features_basic(self, rdb_dataset, target_dataframe, key_mappings):
        """Test basic feature computation with DFS2SQL engine."""
        config = DFSConfig(
            engine="dfs2sql",
            max_depth=1,
            agg_primitives=["count", "mean"],
            use_cutoff_time=False
        )
        
        engine = get_dfs_engine("dfs2sql", config)
        
        # Compute features
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )
        
        # Check that result contains original columns
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
        assert "interaction_time" in result_df.columns
        
        # Check that result has same number of rows
        assert len(result_df) == len(target_dataframe)
        
        # Check that some features were generated
        feature_cols = [col for col in result_df.columns 
                       if col not in target_dataframe.columns]
        assert len(feature_cols) > 0, f"No features generated. Columns: {result_df.columns.tolist()}"


class TestHighLevelAPI:
    """Test the high-level API functions."""
    
    def test_load_rdb(self, test_data_path):
        """Test loading RDB using high-level API."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rdb_path = Path(temp_dir) / "rdb"
            convert_task_dataset_to_rdb(test_data_path, rdb_path)
            
            rdb = load_rdb(str(rdb_path))
            assert isinstance(rdb, RDBDataset)
            assert len(rdb.table_names) == 3
    
    def test_compute_dfs_features_default_config(self, rdb_dataset, target_dataframe, key_mappings):
        """Test computing features with default configuration."""
        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings
        )
        
        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
    
    def test_compute_dfs_features_with_overrides(self, rdb_dataset, target_dataframe, key_mappings):
        """Test computing features with config overrides."""
        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            config_overrides={
                "max_depth": 1,
                "engine": "featuretools",
                "agg_primitives": ["count"]
            }
        )
        
        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns
    
    def test_dfs_pipeline(self, rdb_dataset, target_dataframe, key_mappings):
        """Test DFS pipeline functionality."""
        config = DFSConfig(
            max_depth=1,
            agg_primitives=["count"],
            engine="featuretools"
        )
        
        pipeline = DFSPipeline(
            transform_pipeline=None,  # No transforms for this test
            dfs_config=config
        )
        
        result_df = pipeline.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings
        )
        
        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unknown_engine(self):
        """Test error for unknown engine name."""
        config = DFSConfig(engine="nonexistent")
        
        with pytest.raises(ValueError, match="Unknown DFS engine"):
            get_dfs_engine("nonexistent", config)
    
    def test_empty_target_dataframe(self, rdb_dataset, key_mappings):
        """Test behavior with empty target dataframe."""
        empty_df = pd.DataFrame(columns=["user_id", "item_id"])
        
        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=empty_df,
            key_mappings=key_mappings,
            config_overrides={"max_depth": 1}
        )
        
        # Should return empty dataframe with original columns
        assert len(result_df) == 0
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
    
    def test_invalid_key_mappings(self, rdb_dataset, target_dataframe):
        """Test error for invalid key mappings."""
        invalid_mappings = {
            "user_id": "nonexistent.column"
        }
        
        # This should raise an error when trying to build relationships
        with pytest.raises(Exception):  # Could be various types depending on implementation
            compute_dfs_features(
                rdb=rdb_dataset,
                target_dataframe=target_dataframe,
                key_mappings=invalid_mappings
            )
