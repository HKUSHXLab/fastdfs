"""
Unit tests for shared schema data loading and saving functionality.
"""
import pytest
import numpy as np
import tempfile
import os
from fastdfs.dataset.meta import (
    DBBColumnSchema, DBBTaskMeta, DBBTableSchema, DBBRDBDatasetMeta,
    DBBColumnDType, DBBTaskType, DBBTaskEvalMetric, DBBTableDataFormat
)
from fastdfs.preprocess.transform.base import RDBData, ColumnData
from fastdfs.preprocess.transform_preprocess import RDBTransformPreprocess


class TestSharedSchemaLoading:
    """Test data loading with shared schema design."""
    
    def test_shared_schema_parsing(self):
        """Test parsing of shared_schema field."""
        # Test valid shared_schema format
        shared_schema = "user.user_id"
        table_name, col_name = shared_schema.split('.')
        assert table_name == "user"
        assert col_name == "user_id"
        
        # Test another format
        shared_schema = "interaction.timestamp"
        table_name, col_name = shared_schema.split('.')
        assert table_name == "interaction"
        assert col_name == "timestamp"
    
    def test_column_with_shared_schema(self):
        """Test that columns with shared_schema are handled correctly."""
        col_schema = DBBColumnSchema(
            name="user_id",
            dtype=DBBColumnDType.foreign_key,
            shared_schema="user.user_id",
            capacity=100,
            link_to="user.user_id"
        )
        
        # Create some test data
        col_data = np.array([1, 2, 3, 4, 5])
        column_data = ColumnData(dict(col_schema), col_data)
        
        assert column_data.metadata['shared_schema'] == "user.user_id"
        assert column_data.metadata['dtype'] == DBBColumnDType.foreign_key
        assert np.array_equal(column_data.data, col_data)
    
    def test_column_without_shared_schema(self):
        """Test that columns without shared_schema are handled correctly."""
        col_schema = DBBColumnSchema(
            name="label",
            dtype=DBBColumnDType.category_t,
            num_categories=2
        )
        
        # Create some test data
        col_data = np.array([0, 1, 0, 1, 1])
        column_data = ColumnData(dict(col_schema), col_data)
        
        assert column_data.metadata.get('shared_schema') is None
        assert column_data.metadata['dtype'] == DBBColumnDType.category_t
        assert np.array_equal(column_data.data, col_data)


class TestRDBDataWithSharedSchema:
    """Test RDBData handling with shared schema."""
    
    def test_rdb_data_creation(self):
        """Test creating RDBData with task tables using shared schema."""
        # Create test data for a task table
        task_table_data = {
            "user_id": ColumnData(
                metadata={
                    'name': 'user_id',
                    'dtype': DBBColumnDType.foreign_key,
                    'shared_schema': 'user.user_id',
                    'capacity': 100,
                    'link_to': 'user.user_id'
                },
                data=np.array([1, 2, 3])
            ),
            "label": ColumnData(
                metadata={
                    'name': 'label',
                    'dtype': DBBColumnDType.category_t,
                    'num_categories': 2
                },
                data=np.array([0, 1, 0])
            )
        }
        
        # Create RDBData with task table
        rdb_data = RDBData(
            tables={"__task__:linkpred": task_table_data},
            column_groups=None,
            relationships=None
        )
        
        assert "__task__:linkpred" in rdb_data.tables
        task_table = rdb_data.tables["__task__:linkpred"]
        
        # Check that shared_schema information is preserved
        user_id_col = task_table["user_id"]
        assert user_id_col.metadata['shared_schema'] == 'user.user_id'
        
        # Check that columns without shared_schema work too
        label_col = task_table["label"]
        assert label_col.metadata.get('shared_schema') is None


class TestTaskDataExtraction:
    """Test task data extraction logic with shared schema."""
    
    def test_column_categorization(self):
        """Test that columns are correctly categorized based on shared_schema."""
        # Create column schemas
        columns_with_shared = [
            DBBColumnSchema(
                name="user_id",
                dtype=DBBColumnDType.foreign_key,
                shared_schema="user.user_id",
                capacity=100,
                link_to="user.user_id"
            ),
            DBBColumnSchema(
                name="timestamp",
                dtype=DBBColumnDType.datetime_t,
                shared_schema="interaction.timestamp"
            )
        ]
        
        columns_without_shared = [
            DBBColumnSchema(
                name="label",
                dtype=DBBColumnDType.category_t,
                num_categories=2
            )
        ]
        
        # Test that we can identify which columns have shared schema
        for col in columns_with_shared:
            assert hasattr(col, 'shared_schema')
            assert col.shared_schema is not None
            
        for col in columns_without_shared:
            assert col.shared_schema is None


class TestRelationshipInference:
    """Test relationship inference from shared_schema."""
    
    def test_foreign_key_inference(self):
        """Test inferring foreign key relationships from shared_schema."""
        # Column with shared_schema pointing to a foreign key
        fk_col = DBBColumnSchema(
            name="user_id",
            dtype=DBBColumnDType.foreign_key,
            shared_schema="interaction.user_id",  # Points to FK in interaction table
            capacity=100,
            link_to="user.user_id"
        )
        
        # The shared_schema should help us understand that this task column
        # references the same primary key as interaction.user_id
        shared_table, shared_col = fk_col.shared_schema.split('.')
        assert shared_table == "interaction"
        assert shared_col == "user_id"
        
        # This means we need to look up the relationship for interaction.user_id
        # to find the actual primary key it references
    
    def test_primary_key_inference(self):
        """Test inferring primary key relationships from shared_schema."""
        # Column with shared_schema pointing to a primary key
        pk_col = DBBColumnSchema(
            name="user_id",
            dtype=DBBColumnDType.primary_key,
            shared_schema="user.user_id",  # Points to PK in user table
            capacity=100
        )
        
        # The shared_schema should help us understand that this task column
        # is equivalent to the primary key in the user table
        shared_table, shared_col = pk_col.shared_schema.split('.')
        assert shared_table == "user"
        assert shared_col == "user_id"


if __name__ == "__main__":
    pytest.main([__file__])
