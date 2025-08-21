"""
Tests for Phase 4 API - New table-centric DFS interface.
"""

import pytest
import pandas as pd
from pathlib import Path

from fastdfs.api import load_rdb, compute_dfs_features, DFSPipeline
from fastdfs.dfs import DFSConfig
from fastdfs.transform import RDBTransformPipeline, HandleDummyTable, FeaturizeDatetime, FilterColumn, RDBTransformWrapper


class TestPhase4API:
    """Test the new Phase 4 table-centric API."""
    
    @pytest.fixture
    def test_rdb_path(self):
        """Path to test RDB dataset."""
        return Path(__file__).parent / "data" / "test_rdb_new"
    
    @pytest.fixture
    def sample_target_df(self):
        """Sample target dataframe for testing."""
        return pd.DataFrame({
            "user_id": [1, 2, 3, 1, 2],
            "item_id": [1, 2, 3, 2, 1],
            "timestamp": pd.to_datetime([
                "2023-01-01 10:00:00",
                "2023-01-02 11:00:00", 
                "2023-01-03 12:00:00",
                "2023-01-04 13:00:00",
                "2023-01-05 14:00:00"
            ])
        })
    
    def test_load_rdb(self, test_rdb_path):
        """Test loading RDB dataset."""
        rdb = load_rdb(str(test_rdb_path))
        
        assert rdb.metadata.dataset_name == "sbm_user_item"
        assert len(rdb.table_names) == 3
        assert "user" in rdb.table_names
        assert "item" in rdb.table_names
        assert "interaction" in rdb.table_names
    
    def test_compute_dfs_features_basic(self, test_rdb_path, sample_target_df):
        """Test basic DFS feature computation."""
        rdb = load_rdb(str(test_rdb_path))
        
        features_df = compute_dfs_features(
            rdb=rdb,
            target_dataframe=sample_target_df,
            key_mappings={
                "user_id": "user.user_id",
                "item_id": "item.item_id"
            },
            cutoff_time_column="timestamp",
            config_overrides={
                "max_depth": 2,
                "engine": "featuretools"
            }
        )
        
        # Should have original columns plus generated features
        assert len(features_df.columns) > len(sample_target_df.columns)
        assert len(features_df) == len(sample_target_df)
        
        # Original columns should be preserved
        for col in sample_target_df.columns:
            assert col in features_df.columns
    
    def test_compute_dfs_features_with_config(self, test_rdb_path, sample_target_df):
        """Test DFS feature computation with custom config."""
        rdb = load_rdb(str(test_rdb_path))
        
        config = DFSConfig(
            max_depth=1,
            engine="featuretools",
            agg_primitives=["count", "mean"]
        )
        
        features_df = compute_dfs_features(
            rdb=rdb,
            target_dataframe=sample_target_df,
            key_mappings={
                "user_id": "user.user_id",
                "item_id": "item.item_id"
            },
            cutoff_time_column="timestamp",
            config=config
        )
        
        assert len(features_df.columns) > len(sample_target_df.columns)
        assert len(features_df) == len(sample_target_df)
    
    def test_dfs_pipeline_with_transforms(self, test_rdb_path, sample_target_df):
        """Test DFS pipeline with transform preprocessing."""
        rdb = load_rdb(str(test_rdb_path))
        
        # Create transform pipeline
        transform_pipeline = RDBTransformPipeline([
            HandleDummyTable(),
            RDBTransformWrapper(FeaturizeDatetime(features=["year", "month"])),
            RDBTransformWrapper(FilterColumn(drop_redundant=True))
        ])
        
        # Create DFS pipeline
        pipeline = DFSPipeline(
            transform_pipeline=transform_pipeline,
            dfs_config=DFSConfig(max_depth=2, engine="featuretools")
        )
        
        features_df = pipeline.compute_features(
            rdb=rdb,
            target_dataframe=sample_target_df,
            key_mappings={
                "user_id": "user.user_id",
                "item_id": "item.item_id"
            },
            cutoff_time_column="timestamp"
        )
        
        assert len(features_df.columns) > len(sample_target_df.columns)
        assert len(features_df) == len(sample_target_df)
    
    def test_different_engines_produce_features(self, test_rdb_path, sample_target_df):
        """Test that both engines can generate features."""
        rdb = load_rdb(str(test_rdb_path))
        
        # Test Featuretools engine
        ft_features = compute_dfs_features(
            rdb=rdb,
            target_dataframe=sample_target_df,
            key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
            cutoff_time_column="timestamp",
            config_overrides={"max_depth": 2, "engine": "featuretools"}
        )
        
        # Test DFS2SQL engine  
        sql_features = compute_dfs_features(
            rdb=rdb,
            target_dataframe=sample_target_df,
            key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
            cutoff_time_column="timestamp",
            config_overrides={"max_depth": 2, "engine": "dfs2sql"}
        )
        
        # Both should generate features
        assert len(ft_features.columns) > len(sample_target_df.columns)
        assert len(sql_features.columns) > len(sample_target_df.columns)
        
        # Both should have same number of rows
        assert len(ft_features) == len(sample_target_df)
        assert len(sql_features) == len(sample_target_df)
