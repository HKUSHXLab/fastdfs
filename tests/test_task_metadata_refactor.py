"""
Unit tests for the task table metadata refactor.

Tests the new shared_schema design and removal of target_table field.
"""
import pytest
import numpy as np
import tempfile
import os
import yaml
from fastdfs.dataset.meta import (
    DBBColumnSchema, DBBTaskMeta, DBBTableSchema, DBBRDBDatasetMeta,
    DBBColumnDType, DBBTaskType, DBBTaskEvalMetric, DBBTableDataFormat
)
from fastdfs.preprocess.transform.base import (
    make_task_table_name, unmake_task_table_name, is_task_table
)


class TestTaskTableNaming:
    """Test the updated task table naming functions."""
    
    def test_make_task_table_name(self):
        """Test that task table names are created correctly."""
        task_name = "linkpred"
        result = make_task_table_name(task_name)
        assert result == "__task__:linkpred"
    
    def test_unmake_task_table_name(self):
        """Test that task table names are parsed correctly."""
        task_table_name = "__task__:linkpred"
        result = unmake_task_table_name(task_table_name)
        assert result == "linkpred"
    
    def test_is_task_table(self):
        """Test task table detection."""
        assert is_task_table("__task__:linkpred")
        assert not is_task_table("user")
        assert not is_task_table("interaction")


class TestTaskMetadataSchema:
    """Test the new task metadata schema."""
    
    def test_column_schema_with_shared_schema(self):
        """Test that column schema can have shared_schema field."""
        col_schema = DBBColumnSchema(
            name="user_id",
            dtype=DBBColumnDType.foreign_key,
            shared_schema="user.user_id",
            capacity=100,
            link_to="user.user_id"
        )
        assert col_schema.name == "user_id"
        assert col_schema.dtype == DBBColumnDType.foreign_key
        assert col_schema.shared_schema == "user.user_id"
        assert col_schema.capacity == 100
    
    def test_column_schema_without_shared_schema(self):
        """Test that column schema works without shared_schema field."""
        col_schema = DBBColumnSchema(
            name="label",
            dtype=DBBColumnDType.category_t,
            num_categories=2
        )
        assert col_schema.name == "label"
        assert col_schema.dtype == DBBColumnDType.category_t
        assert col_schema.shared_schema is None
    
    def test_task_meta_without_target_table(self):
        """Test that task metadata works without target_table field."""
        columns = [
            DBBColumnSchema(
                name="user_id",
                dtype=DBBColumnDType.foreign_key,
                shared_schema="user.user_id",
                capacity=100,
                link_to="user.user_id"
            ),
            DBBColumnSchema(
                name="label",
                dtype=DBBColumnDType.category_t,
                num_categories=2
            )
        ]
        
        task_meta = DBBTaskMeta(
            name="linkpred",
            source="linkpred/{split}.npz",
            format=DBBTableDataFormat.NUMPY,
            columns=columns,
            evaluation_metric=DBBTaskEvalMetric.auroc,
            target_column="label",
            task_type=DBBTaskType.classification
        )
        
        assert task_meta.name == "linkpred"
        assert task_meta.target_column == "label"
        assert task_meta.task_type == DBBTaskType.classification
        assert not hasattr(task_meta, 'target_table')
        assert not hasattr(task_meta, 'key_prediction_label_column')
        assert not hasattr(task_meta, 'key_prediction_query_idx_column')


class TestMetadataYamlFormat:
    """Test that the new metadata format can be loaded from YAML."""
    
    def test_load_new_metadata_format(self):
        """Test loading metadata with new format."""
        metadata_dict = {
            'dataset_name': 'test_dataset',
            'tables': [
                {
                    'name': 'user',
                    'source': 'data/user.npz',
                    'format': 'numpy',
                    'time_column': None,
                    'columns': [
                        {
                            'name': 'user_id',
                            'dtype': 'primary_key',
                            'capacity': 100
                        }
                    ]
                }
            ],
            'tasks': [
                {
                    'name': 'linkpred',
                    'source': 'linkpred/{split}.npz',
                    'format': 'numpy',
                    'columns': [
                        {
                            'name': 'user_id',
                            'dtype': 'foreign_key',
                            'shared_schema': 'user.user_id',
                            'capacity': 100,
                            'link_to': 'user.user_id'
                        },
                        {
                            'name': 'label',
                            'dtype': 'category',
                            'num_categories': 2
                        }
                    ],
                    'evaluation_metric': 'auroc',
                    'target_column': 'label',
                    'task_type': 'classification'
                }
            ]
        }
        
        # Test that we can create the metadata object
        dataset_meta = DBBRDBDatasetMeta(**metadata_dict)
        assert dataset_meta.dataset_name == 'test_dataset'
        assert len(dataset_meta.tables) == 1
        
        # Check task metadata
        task = dataset_meta.tasks[0]
        assert task.name == 'linkpred'
        assert task.target_column == 'label'
        assert task.task_type == DBBTaskType.classification
        
        # Check columns with shared_schema
        user_id_col = task.columns[0]
        assert user_id_col.name == 'user_id'
        assert user_id_col.shared_schema == 'user.user_id'
        
        # Check columns without shared_schema
        label_col = task.columns[1]
        assert label_col.name == 'label'
        assert label_col.shared_schema is None
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata in new format."""
        # Create metadata with new format
        user_table = DBBTableSchema(
            name='user',
            source='data/user.npz',
            format=DBBTableDataFormat.NUMPY,
            time_column=None,
            columns=[
                DBBColumnSchema(
                    name='user_id',
                    dtype=DBBColumnDType.primary_key,
                    capacity=100
                )
            ]
        )
        
        task_meta = DBBTaskMeta(
            name='linkpred',
            source='linkpred/{split}.npz',
            format=DBBTableDataFormat.NUMPY,
            columns=[
                DBBColumnSchema(
                    name='user_id',
                    dtype=DBBColumnDType.foreign_key,
                    shared_schema='user.user_id',
                    capacity=100,
                    link_to='user.user_id'
                ),
                DBBColumnSchema(
                    name='label',
                    dtype=DBBColumnDType.category_t,
                    num_categories=2
                )
            ],
            evaluation_metric=DBBTaskEvalMetric.auroc,
            target_column='label',
            task_type=DBBTaskType.classification
        )
        
        dataset_meta = DBBRDBDatasetMeta(
            dataset_name='test_dataset',
            tables=[user_table],
            tasks=[task_meta]
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(dataset_meta.dict(), f)
            temp_file = f.name
        
        try:
            # Load from file
            with open(temp_file, 'r') as f:
                loaded_dict = yaml.safe_load(f)
            
            loaded_meta = DBBRDBDatasetMeta(**loaded_dict)
            
            # Verify the loaded metadata
            assert loaded_meta.dataset_name == 'test_dataset'
            assert len(loaded_meta.tasks) == 1
            
            loaded_task = loaded_meta.tasks[0]
            assert loaded_task.name == 'linkpred'
            assert loaded_task.target_column == 'label'
            assert not hasattr(loaded_task, 'target_table')
            
            # Check shared_schema field
            user_id_col = loaded_task.columns[0]
            assert user_id_col.shared_schema == 'user.user_id'
            
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
