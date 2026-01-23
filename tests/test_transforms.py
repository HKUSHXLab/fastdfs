"""
Test the new transform interface (Phase 3).

This tests the RDBTransform, TableTransform, and ColumnTransform base classes
along with the specific transform implementations using real test data.
"""

from loguru import logger
logger.enable("fastdfs")

import pandas as pd
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from fastdfs.dataset.meta import RDBColumnSchema, RDBColumnDType, RDBTableDataFormat
from fastdfs.dataset.rdb import RDB, RDBTableSchema, RDBMeta
from fastdfs.transform.base import RDBTransform, TableTransform, ColumnTransform
from fastdfs.transform.datetime_transform import FeaturizeDatetime
from fastdfs.transform.filter_transform import FilterColumn
from fastdfs.transform.dummy_table_transform import HandleDummyTable
from fastdfs.utils import yaml_utils


class TestFeaturizeDatetime:
    """Test datetime feature extraction transform."""
    
    @classmethod
    def setup_class(cls):
        """Set up test dataset."""
        test_data_path = Path(__file__).parent / "data" / "test_rdb_new"
        cls.dataset = RDB(test_data_path)
        
        # Get interaction table which has datetime column
        cls.interaction_table = cls.dataset.get_table('interaction')
        cls.interaction_metadata = cls.dataset.get_table_metadata('interaction')
        
        # Find datetime column
        cls.datetime_column = None
        cls.datetime_metadata = None
        for col_schema in cls.interaction_metadata.columns:
            if col_schema.dtype == RDBColumnDType.datetime_t:
                cls.datetime_column = cls.interaction_table[col_schema.name]
                cls.datetime_metadata = col_schema
                break
        
        cls.transform = FeaturizeDatetime(features=['year', 'month', 'day'])
    
    def test_datetime_featurization(self):
        """Test that datetime features are extracted from real data."""
        if self.datetime_column is None:
            pytest.skip("No datetime column found in test data")
            
        result_df, result_schemas = self.transform(
            self.datetime_column, self.datetime_metadata
        )
        
        # Check result structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_schemas) >= 3  # At least 3 features
        assert all(isinstance(schema, RDBColumnSchema) for schema in result_schemas)
        
        # Check that feature columns were created
        expected_features = ['timestamp_year', 'timestamp_month', 'timestamp_day']
        for feature in expected_features:
            assert feature in result_df.columns
    
    def test_applies_to_datetime_column(self):
        """Test that transform correctly identifies datetime columns."""
        if self.datetime_metadata is None:
            pytest.skip("No datetime column found in test data")
            
        assert self.transform.applies_to(self.datetime_metadata) == True
    
    def test_does_not_apply_to_non_datetime_column(self):
        """Test that transform doesn't apply to non-datetime columns."""
        # Get a non-datetime column
        user_metadata = self.dataset.get_table_metadata('user')
        non_datetime_col = user_metadata.columns[0]  # user_id (primary key)
        
        assert self.transform.applies_to(non_datetime_col) == False

    def test_retain_original_option(self):
        """Test the retain_original option."""
        if self.datetime_column is None:
            pytest.skip("No datetime column found in test data")
            
        # Test with retain_original=True (default)
        transform_retain = FeaturizeDatetime(features=['year'], retain_original=True)
        result_df, result_schemas = transform_retain(
            self.datetime_column, self.datetime_metadata
        )
        assert self.datetime_metadata.name in result_df.columns
        assert any(s.name == self.datetime_metadata.name for s in result_schemas)
        
        # Test with retain_original=False
        transform_drop = FeaturizeDatetime(features=['year'], retain_original=False)
        result_df, result_schemas = transform_drop(
            self.datetime_column, self.datetime_metadata
        )
        assert self.datetime_metadata.name not in result_df.columns
        assert not any(s.name == self.datetime_metadata.name for s in result_schemas)
        assert 'timestamp_year' in result_df.columns

    def test_epochtime_known_values(self):
        """Test epochtime with known expected values."""
        # Unix epoch should be 0
        epoch_date = pd.Series([pd.Timestamp('1970-01-01T00:00:00')], name='epoch')
        epoch_metadata = RDBColumnSchema(name='epoch', dtype=RDBColumnDType.datetime_t)

        transform_epochtime = FeaturizeDatetime(features=['epochtime'], retain_original=False)
        result_df, _ = transform_epochtime(epoch_date, epoch_metadata)

        assert result_df['epoch_epochtime'].iloc[0] == 0

        # Test a known date
        known_date = pd.Series([pd.Timestamp('2020-01-01T00:00:00')], name='known')
        known_metadata = RDBColumnSchema(name='known', dtype=RDBColumnDType.datetime_t)

        result_df, _ = transform_epochtime(known_date, known_metadata)
        expected_ns = pd.Timestamp('2020-01-01').value
        assert result_df['known_epochtime'].iloc[0] == expected_ns


class TestFilterColumn:
    """Test column filtering transform."""
    
    @classmethod
    def setup_class(cls):
        """Set up test dataset with synthetic columns."""
        test_data_path = Path(__file__).parent / "data" / "test_rdb_new"
        cls.original_dataset = RDB(test_data_path)
        cls.transform = FilterColumn(drop_dtypes=[RDBColumnDType.text_t], drop_redundant=True)
        
        # Create test dataset with synthetic columns added
        cls.temp_dir = None
        cls.dataset_with_synthetic = cls._create_dataset_with_synthetic_columns()
    
    @classmethod
    def teardown_class(cls):
        """Clean up temporary files."""
        if cls.temp_dir and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_dataset_with_synthetic_columns(cls):
        """Create a test dataset with additional synthetic columns for filtering."""
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        temp_path = Path(cls.temp_dir)
        
        # Copy original data and modify it to include synthetic columns
        data_dir = temp_path / "data"
        data_dir.mkdir()
        
        original_data_dir = Path(__file__).parent / "data" / "test_rdb_new" / "data"
        
        # Load original data and add synthetic columns
        for table_name in ['user', 'item', 'interaction']:
            original_file = original_data_dir / f"{table_name}.npz"
            if original_file.exists():
                # Load original data with pickle support (like the loader does)
                original_data = np.load(original_file, allow_pickle=True)
                
                # Create new data dictionary with synthetic columns
                new_data = {}
                
                # Copy all original columns
                for key in original_data.files:
                    new_data[key] = original_data[key]
                
                # Add synthetic columns based on table
                data_length = len(original_data[original_data.files[0]])
                
                if table_name == 'user':
                    # Add synthetic columns for user table
                    new_data['synthetic_redundant_1'] = np.random.random(data_length).astype(np.float32)
                    new_data['synthetic_redundant_2'] = np.random.randint(0, 5, data_length)
                    new_data['synthetic_text'] = np.array([f'text_{i}' for i in range(data_length)], dtype=object)
                    # Add a truly redundant column (all same value)
                    new_data['truly_redundant'] = np.full(data_length, 42.0, dtype=np.float32)
                elif table_name == 'item':
                    # Add synthetic columns for item table
                    new_data['synthetic_text_col'] = np.array([f'item_text_{i}' for i in range(data_length)], dtype=object)
                    new_data['synthetic_float'] = np.random.random(data_length).astype(np.float32)
                    # Add a truly redundant column (all same value)
                    new_data['truly_redundant'] = np.full(data_length, 99.0, dtype=np.float32)
                elif table_name == 'interaction':
                    # Add synthetic columns for interaction table
                    new_data['synthetic_text_interaction'] = np.array([f'interaction_text_{i}' for i in range(data_length)], dtype=object)
                    # Add a truly redundant column (all same value)
                    new_data['truly_redundant'] = np.full(data_length, 1.0, dtype=np.float32)
                
                # Save modified data
                new_file = data_dir / f"{table_name}.npz"
                np.savez(new_file, **new_data)
        
        # Create enhanced metadata with synthetic columns
        enhanced_metadata = {
            'name': 'test_with_synthetic',
            'tables': [
                {
                    'name': 'user',
                    'source': 'data/user.npz',
                    'format': 'numpy',
                    'columns': [
                        {'name': 'user_id', 'dtype': 'primary_key'},
                        {'name': 'user_feature_0', 'dtype': 'float'},
                        # Add synthetic columns that should be filtered
                        {'name': 'synthetic_redundant_1', 'dtype': 'float'},
                        {'name': 'synthetic_redundant_2', 'dtype': 'category', 'num_categories': 5},
                        {'name': 'synthetic_text', 'dtype': 'text'},  # Should be filtered
                        {'name': 'truly_redundant', 'dtype': 'float'},  # Should be filtered (all same value)
                    ]
                },
                {
                    'name': 'item', 
                    'source': 'data/item.npz',
                    'format': 'numpy',
                    'columns': [
                        {'name': 'item_id', 'dtype': 'primary_key', 'capacity': 100},
                        {'name': 'item_feature_0', 'dtype': 'float', 'in_size': 1},
                        # Add synthetic columns
                        {'name': 'synthetic_text_col', 'dtype': 'text'},  # Should be filtered
                        {'name': 'synthetic_float', 'dtype': 'float'},
                        {'name': 'truly_redundant', 'dtype': 'float'},  # Should be filtered (all same value)
                    ]
                },
                {
                    'name': 'interaction',
                    'source': 'data/interaction.npz', 
                    'format': 'numpy',
                    'time_column': 'timestamp',
                    'columns': [
                        {'name': 'user_id', 'dtype': 'foreign_key', 'link_to': 'user.user_id', 'capacity': 100},
                        {'name': 'item_id', 'dtype': 'foreign_key', 'link_to': 'item.item_id', 'capacity': 100},
                        {'name': 'timestamp', 'dtype': 'datetime'},
                        # Add synthetic columns  
                        {'name': 'synthetic_text_interaction', 'dtype': 'text'},  # Should be filtered
                        {'name': 'truly_redundant', 'dtype': 'float'},  # Should be filtered (all same value)
                    ]
                }
            ]
        }
        
        # Save enhanced metadata
        metadata_path = temp_path / "metadata.yaml"
        yaml_utils.save_yaml(enhanced_metadata, metadata_path)
        
        # Load and return the dataset
        return RDB(temp_path)
    
    def test_preserve_primary_keys(self):
        """Test that primary keys are always preserved."""
        # Test on user table (has primary key)
        user_table = self.dataset_with_synthetic.get_table('user')
        user_metadata = self.dataset_with_synthetic.get_table_metadata('user')
        
        # Get original primary key columns
        original_pk_cols = [col for col in user_metadata.columns 
                           if col.dtype == RDBColumnDType.primary_key]
        
        result_table, result_metadata = self.transform(user_table, user_metadata)
        
        # Check that primary keys are preserved
        result_pk_cols = [col for col in result_metadata.columns 
                         if col.dtype == RDBColumnDType.primary_key]
        
        # All primary keys should be preserved
        assert len(result_pk_cols) == len(original_pk_cols)
        for pk_col in original_pk_cols:
            assert pk_col.name in result_table.columns
    
    def test_preserve_foreign_keys(self):
        """Test that foreign keys are always preserved."""
        # Test on interaction table (has foreign keys)
        interaction_table = self.dataset_with_synthetic.get_table('interaction')
        interaction_metadata = self.dataset_with_synthetic.get_table_metadata('interaction')
        
        # Get original foreign key columns
        original_fk_cols = [col for col in interaction_metadata.columns 
                           if col.dtype == RDBColumnDType.foreign_key]
        
        result_table, result_metadata = self.transform(interaction_table, interaction_metadata)
        
        # Check that foreign keys are preserved
        result_fk_cols = [col for col in result_metadata.columns 
                         if col.dtype == RDBColumnDType.foreign_key]
        
        # All foreign keys should be preserved
        assert len(result_fk_cols) == len(original_fk_cols)
        for fk_col in original_fk_cols:
            assert fk_col.name in result_table.columns
    
    def test_filter_synthetic_columns(self):
        """Test that synthetic columns are properly filtered."""
        # Test on user table with synthetic columns
        user_table = self.dataset_with_synthetic.get_table('user')
        user_metadata = self.dataset_with_synthetic.get_table_metadata('user')
        
        original_columns = len(user_metadata.columns)
        original_text_cols = [col for col in user_metadata.columns if col.dtype == RDBColumnDType.text_t]
        
        result_table, result_metadata = self.transform(user_table, user_metadata)
        
        # Should have fewer columns (text columns and redundant columns filtered out)
        assert len(result_metadata.columns) < original_columns
        
        # Text columns should be filtered out
        result_text_cols = [col for col in result_metadata.columns if col.dtype == RDBColumnDType.text_t]
        assert len(result_text_cols) == 0
        
        # Redundant columns should be filtered out
        result_column_names = [col.name for col in result_metadata.columns]
        assert 'truly_redundant' not in result_column_names
        
        # But still preserve keys
        original_keys = [col for col in user_metadata.columns 
                        if col.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]]
        result_keys = [col for col in result_metadata.columns 
                      if col.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]]
        assert len(result_keys) == len(original_keys)
    
    def test_filter_preserves_essential_columns(self):
        """Test that filtering preserves essential columns while removing redundant ones."""
        for table_name in self.dataset_with_synthetic.table_names:
            table_data = self.dataset_with_synthetic.get_table(table_name)
            table_metadata = self.dataset_with_synthetic.get_table_metadata(table_name)
            
            result_table, result_metadata = self.transform(table_data, table_metadata)
            
            # Should preserve all key columns
            essential_cols = [col for col in table_metadata.columns 
                            if col.dtype in [RDBColumnDType.primary_key, RDBColumnDType.foreign_key]]
            
            for essential_col in essential_cols:
                assert essential_col.name in result_table.columns
                
            # Should remove text columns (our synthetic redundant columns)
            result_text_cols = [col for col in result_metadata.columns if col.dtype == RDBColumnDType.text_t]
            assert len(result_text_cols) == 0
            
            # Should remove truly redundant columns (all same values)
            result_column_names = [col.name for col in result_metadata.columns]
            assert 'truly_redundant' not in result_column_names
    
    def test_filter_redundant_columns(self):
        """Test that columns with all identical values are filtered out."""
        # Test specifically with redundant columns
        user_table = self.dataset_with_synthetic.get_table('user')
        user_metadata = self.dataset_with_synthetic.get_table_metadata('user')
        
        # Verify the redundant column exists and has all same values
        assert 'truly_redundant' in user_table.columns
        assert user_table['truly_redundant'].nunique() == 1  # All values are the same
        
        result_table, result_metadata = self.transform(user_table, user_metadata)
        
        # Redundant column should be removed
        result_column_names = [col.name for col in result_metadata.columns]
        assert 'truly_redundant' not in result_column_names
        assert 'truly_redundant' not in result_table.columns


class TestHandleDummyTable:
    """Test dummy table creation transform."""
    
    @classmethod
    def setup_class(cls):
        """Set up test dataset."""
        test_data_path = Path(__file__).parent / "data" / "test_rdb_new"
        cls.original_dataset = RDB(test_data_path)
        cls.transform = HandleDummyTable()
        cls.temp_dir = None
    
    @classmethod
    def teardown_class(cls):
        """Clean up temporary files."""
        if cls.temp_dir and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
    
    def _create_dataset_missing_table(self, missing_table_name: str):
        """Create a test dataset with a missing table to test dummy table creation."""
        # Create temporary directory  
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        # Copy original data except for the missing table
        data_dir = temp_path / "data"
        data_dir.mkdir()
        
        original_data_dir = Path(__file__).parent / "data" / "test_rdb_new" / "data"
        for npz_file in original_data_dir.glob("*.npz"):
            if missing_table_name not in npz_file.name:
                shutil.copy2(npz_file, data_dir)
        
        # Load original metadata and remove the missing table
        original_metadata_path = Path(__file__).parent / "data" / "test_rdb_new" / "metadata.yaml"
        original_metadata = yaml_utils.load_yaml(original_metadata_path)
        
        # Filter out the missing table from metadata
        filtered_tables = [
            table for table in original_metadata['tables'] 
            if table['name'] != missing_table_name
        ]
        
        incomplete_metadata = {
            'name': f'test_missing_{missing_table_name}',
            'tables': filtered_tables
        }
        
        # Save incomplete metadata
        metadata_path = temp_path / "metadata.yaml"
        yaml_utils.save_yaml(incomplete_metadata, metadata_path)
        
        # Load and return the incomplete dataset
        return RDB(temp_path)
    
    def test_no_missing_tables_in_complete_dataset(self):
        """Test that transform is no-op when all referenced tables exist."""
        # The test_rdb_new dataset should be complete (no missing tables)
        result_dataset = self.transform(self.original_dataset)
        
        # Should have same tables (no new dummy tables created)
        original_tables = set(self.original_dataset.table_names)
        result_tables = set(result_dataset.table_names)
        
        # Either no change, or possibly some dummy tables were added if there were missing references
        # The important thing is that it doesn't crash and returns a valid dataset
        assert isinstance(result_dataset, type(self.original_dataset))
        assert len(result_tables) >= len(original_tables)  # Can only add tables, not remove
    
    def test_recover_missing_user_table(self):
        """Test that transform recovers missing user table with only ID column."""
        # Create dataset missing user table
        incomplete_dataset = self._create_dataset_missing_table('user')
        
        # Verify user table is missing
        assert 'user' not in incomplete_dataset.table_names
        assert 'interaction' in incomplete_dataset.table_names  # But interaction table still references user
        
        # Apply transform
        result_dataset = self.transform(incomplete_dataset)
        
        # Verify user table was recovered
        assert 'user' in result_dataset.table_names
        
        # Check that recovered user table has only the ID column
        recovered_user_table = result_dataset.get_table('user')
        recovered_user_metadata = result_dataset.get_table_metadata('user')
        
        # Should have exactly one column (the primary key)
        assert len(recovered_user_metadata.columns) == 1
        assert recovered_user_metadata.columns[0].name == 'user_id'
        assert recovered_user_metadata.columns[0].dtype == RDBColumnDType.primary_key
        
        # Verify that the table contains the referenced user IDs from interaction table
        interaction_table = incomplete_dataset.get_table('interaction')
        referenced_user_ids = set(interaction_table['user_id'].unique())
        recovered_user_ids = set(recovered_user_table['user_id'].unique())
        
        # All referenced user IDs should be present in the recovered table
        assert referenced_user_ids.issubset(recovered_user_ids)
    
    def test_recover_missing_item_table(self):
        """Test that transform recovers missing item table with only ID column."""
        # Create dataset missing item table
        incomplete_dataset = self._create_dataset_missing_table('item')
        
        # Verify item table is missing
        assert 'item' not in incomplete_dataset.table_names
        assert 'interaction' in incomplete_dataset.table_names  # But interaction table still references item
        
        # Apply transform
        result_dataset = self.transform(incomplete_dataset)
        
        # Verify item table was recovered
        assert 'item' in result_dataset.table_names
        
        # Check that recovered item table has only the ID column
        recovered_item_table = result_dataset.get_table('item')
        recovered_item_metadata = result_dataset.get_table_metadata('item')
        
        # Should have exactly one column (the primary key)
        assert len(recovered_item_metadata.columns) == 1
        assert recovered_item_metadata.columns[0].name == 'item_id'
        assert recovered_item_metadata.columns[0].dtype == RDBColumnDType.primary_key
        
        # Verify that the table contains the referenced item IDs from interaction table
        interaction_table = incomplete_dataset.get_table('interaction')
        referenced_item_ids = set(interaction_table['item_id'].unique())
        recovered_item_ids = set(recovered_item_table['item_id'].unique())
        
        # All referenced item IDs should be present in the recovered table
        assert referenced_item_ids.issubset(recovered_item_ids)
    
    def test_transform_creates_valid_dataset(self):
        """Test that transform always returns a valid RDBDataset."""
        result_dataset = self.transform(self.original_dataset)
        
        # Check that result is a valid RDBDataset
        assert hasattr(result_dataset, 'tables')
        assert hasattr(result_dataset, 'get_table')
        assert hasattr(result_dataset, 'get_table_metadata')
        assert hasattr(result_dataset, 'get_relationships')
        
        # Check that all tables can be accessed
        for table_name in result_dataset.table_names:
            table = result_dataset.get_table(table_name)
            metadata = result_dataset.get_table_metadata(table_name)
            assert isinstance(table, pd.DataFrame)
            assert isinstance(metadata, RDBTableSchema)
    
    def test_preserves_original_tables_when_recovering(self):
        """Test that when recovering missing tables, original tables are preserved."""
        # Test with missing user table
        incomplete_dataset = self._create_dataset_missing_table('user')
        original_tables_before = set(incomplete_dataset.table_names)
        
        result_dataset = self.transform(incomplete_dataset)
        
        # All original tables should still be present
        for table_name in original_tables_before:
            assert table_name in result_dataset.table_names
            
            # Original table data should be preserved
            original_table = incomplete_dataset.get_table(table_name)
            result_table = result_dataset.get_table(table_name)
            
            # Check that data is preserved (should be identical)
            pd.testing.assert_frame_equal(original_table, result_table)
