"""
Tests for the new simplified RDB dataset implementation (Phase 1).

This test file validates that the new RDBDataset class correctly loads
and provides access to relational database tables without tasks.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from fastdfs.dataset.rdb import (
    RDBDataset, 
    convert_task_dataset_to_rdb,
    extract_target_tables_from_tasks
)
from fastdfs.dataset.meta import DBBColumnDType, DBBTableDataFormat


class TestRDBDataset:
    """Test suite for the new simplified RDB dataset implementation."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to existing test dataset."""
        return Path(__file__).parent / "data" / "test_rdb"
    
    @pytest.fixture 
    def rdb_dataset(self, test_data_path):
        """Create RDB dataset from existing test data by converting it first."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rdb_path = Path(temp_dir) / "rdb"
            
            # Convert task-based dataset to RDB-only format
            convert_task_dataset_to_rdb(test_data_path, rdb_path)
            
            # Load the converted RDB dataset
            yield RDBDataset(rdb_path)
    
    def test_load_rdb_dataset(self, rdb_dataset):
        """Test that RDB dataset loads correctly."""
        assert rdb_dataset.metadata.dataset_name == "sbm_user_item"
        assert len(rdb_dataset.table_names) == 3
        assert set(rdb_dataset.table_names) == {"user", "item", "interaction"}
    
    def test_get_table(self, rdb_dataset):
        """Test getting individual tables as DataFrames."""
        # Test user table
        user_df = rdb_dataset.get_table("user")
        assert isinstance(user_df, pd.DataFrame)
        assert "user_id" in user_df.columns
        assert "user_feature_0" in user_df.columns
        assert len(user_df) > 0
        
        # Test item table
        item_df = rdb_dataset.get_table("item")
        assert isinstance(item_df, pd.DataFrame)
        assert "item_id" in item_df.columns
        assert "item_feature_0" in item_df.columns
        assert len(item_df) > 0
        
        # Test interaction table
        interaction_df = rdb_dataset.get_table("interaction")
        assert isinstance(interaction_df, pd.DataFrame)
        assert "user_id" in interaction_df.columns
        assert "item_id" in interaction_df.columns
        assert "timestamp" in interaction_df.columns
        assert len(interaction_df) > 0
    
    def test_get_table_metadata(self, rdb_dataset):
        """Test getting table metadata."""
        user_meta = rdb_dataset.get_table_metadata("user")
        assert user_meta.name == "user"
        assert user_meta.source == "data/user.npz"
        assert user_meta.format == DBBTableDataFormat.NUMPY
        assert len(user_meta.columns) == 2
        
        # Check column metadata
        user_id_col = None
        for col in user_meta.columns:
            if col.name == "user_id":
                user_id_col = col
                break
        
        assert user_id_col is not None
        assert user_id_col.dtype == DBBColumnDType.primary_key
    
    def test_get_relationships(self, rdb_dataset):
        """Test extracting relationships from foreign keys."""
        relationships = rdb_dataset.get_relationships()
        
        # Should have 2 relationships: interaction -> user, interaction -> item
        assert len(relationships) == 2
        
        relationship_tuples = set(relationships)
        expected_relationships = {
            ("interaction", "user_id", "user", "user_id"),
            ("interaction", "item_id", "item", "item_id")
        }
        assert relationship_tuples == expected_relationships
    
    def test_create_new_with_tables(self, rdb_dataset):
        """Test creating new dataset with modified tables."""
        # Get original user table
        original_user_df = rdb_dataset.get_table("user")
        
        # Create modified tables
        modified_user_df = original_user_df.copy()
        modified_user_df["new_feature"] = 42.0
        
        new_tables = {
            "user": modified_user_df,
            "item": rdb_dataset.get_table("item"),
            "interaction": rdb_dataset.get_table("interaction")
        }
        
        # Create new dataset
        new_dataset = rdb_dataset.create_new_with_tables(new_tables)
        
        # Test that new dataset has modified table
        new_user_df = new_dataset.get_table("user")
        assert "new_feature" in new_user_df.columns
        assert (new_user_df["new_feature"] == 42.0).all()
        
        # Test that original dataset is unchanged
        original_user_df_after = rdb_dataset.get_table("user")
        assert "new_feature" not in original_user_df_after.columns
    
    def test_table_not_found_error(self, rdb_dataset):
        """Test error handling for non-existent tables."""
        with pytest.raises(ValueError, match="Table nonexistent not found"):
            rdb_dataset.get_table("nonexistent")
        
        with pytest.raises(ValueError, match="Table nonexistent not found"):
            rdb_dataset.get_table_metadata("nonexistent")
    
    def test_sqlalchemy_metadata(self, rdb_dataset):
        """Test SQLAlchemy metadata generation."""
        metadata = rdb_dataset.sqlalchemy_metadata
        
        # Check that all tables are present
        table_names = set(metadata.tables.keys())
        assert table_names == {"user", "item", "interaction"}
        
        # Check that tables have expected columns
        user_table = metadata.tables["user"]
        assert "user_id" in [col.name for col in user_table.columns]
        assert "user_feature_0" in [col.name for col in user_table.columns]
        
        interaction_table = metadata.tables["interaction"]
        assert "user_id" in [col.name for col in interaction_table.columns]
        assert "item_id" in [col.name for col in interaction_table.columns]
        assert "timestamp" in [col.name for col in interaction_table.columns]


class TestDatasetMigration:
    """Test suite for migration utilities."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to existing test dataset."""
        return Path(__file__).parent / "data" / "test_rdb"
    
    def test_convert_task_dataset_to_rdb(self, test_data_path):
        """Test converting task-based dataset to RDB format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rdb_output_path = Path(temp_dir) / "converted_rdb"
            
            # Convert dataset
            convert_task_dataset_to_rdb(test_data_path, rdb_output_path)
            
            # Check that metadata was created
            metadata_path = rdb_output_path / "metadata.yaml"
            assert metadata_path.exists()
            
            # Check that table data files were copied
            user_data_path = rdb_output_path / "data" / "user.npz"
            assert user_data_path.exists()
            
            item_data_path = rdb_output_path / "data" / "item.npz"
            assert item_data_path.exists()
            
            interaction_data_path = rdb_output_path / "data" / "interaction.npz"
            assert interaction_data_path.exists()
            
            # Test that converted dataset can be loaded
            rdb_dataset = RDBDataset(rdb_output_path)
            assert rdb_dataset.metadata.dataset_name == "sbm_user_item"
            assert len(rdb_dataset.table_names) == 3
    
    def test_extract_target_tables_from_tasks(self, test_data_path):
        """Test extracting task data as target tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "target_tables"
            
            # Extract target tables
            extract_target_tables_from_tasks(test_data_path, output_dir)
            
            # Check that target table files were created
            train_file = output_dir / "linkpred_train.parquet"
            assert train_file.exists()
            
            validation_file = output_dir / "linkpred_validation.parquet"
            assert validation_file.exists()
            
            test_file = output_dir / "linkpred_test.parquet"
            assert test_file.exists()
            
            # Test that target table can be loaded
            train_df = pd.read_parquet(train_file)
            assert "user_id" in train_df.columns
            assert "item_id" in train_df.columns
            assert "timestamp" in train_df.columns
            assert "label" in train_df.columns
            assert len(train_df) > 0


class TestRDBDatasetEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to existing test dataset."""
        return Path(__file__).parent / "data" / "test_rdb"
    
    def test_missing_metadata_file(self):
        """Test error when metadata file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "empty"
            empty_path.mkdir()
            
            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                RDBDataset(empty_path)
    
    def test_missing_data_file(self):
        """Test error when table data file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "incomplete"
            dataset_path.mkdir()
            
            # Create metadata with reference to non-existent data file
            metadata = {
                "dataset_name": "test",
                "tables": [
                    {
                        "name": "test_table",
                        "source": "nonexistent.npz",
                        "format": "numpy",
                        "columns": [
                            {"name": "id", "dtype": "primary_key"}
                        ]
                    }
                ]
            }
            
            import yaml
            with open(dataset_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)
            
            with pytest.raises(FileNotFoundError, match="Table data file not found"):
                RDBDataset(dataset_path)
    
    def test_missing_column_in_data(self, test_data_path):
        """Test error when metadata references column not in data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "bad_metadata"
            dataset_path.mkdir()
            
            # Copy data files
            data_dir = dataset_path / "data"
            data_dir.mkdir()
            shutil.copy2(test_data_path / "data" / "user.npz", data_dir / "user.npz")
            
            # Create metadata with reference to non-existent column
            metadata = {
                "dataset_name": "test",
                "tables": [
                    {
                        "name": "user",
                        "source": "data/user.npz", 
                        "format": "numpy",
                        "columns": [
                            {"name": "user_id", "dtype": "primary_key"},
                            {"name": "nonexistent_column", "dtype": "float"}
                        ]
                    }
                ]
            }
            
            import yaml
            with open(dataset_path / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f)
            
            with pytest.raises(ValueError, match="Column nonexistent_column not found"):
                RDBDataset(dataset_path)
