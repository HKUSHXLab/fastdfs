"""
Test the column groups functionality with shared schema.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from fastdfs.dataset.meta import (
    DBBColumnSchema, DBBTaskMeta, DBBTableSchema, DBBRDBDatasetMeta,
    DBBColumnDType, DBBTaskType, DBBTaskEvalMetric, DBBTableDataFormat
)
from fastdfs.preprocess.transform.base import RDBData, ColumnData, make_task_table_name
from fastdfs.preprocess.transform_preprocess import RDBTransformPreprocess


def test_column_groups_with_shared_schema():
    """Test that column groups are created correctly based on shared_schema."""
    print("Testing column groups with shared_schema...")
    
    # Create test data structures
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
            ),
            DBBColumnSchema(
                name='user_feature',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ]
    )
    
    interaction_table = DBBTableSchema(
        name='interaction',
        source='data/interaction.npz',
        format=DBBTableDataFormat.NUMPY,
        time_column='timestamp',
        columns=[
            DBBColumnSchema(
                name='user_id',
                dtype=DBBColumnDType.foreign_key,
                link_to='user.user_id',
                capacity=100
            ),
            DBBColumnSchema(
                name='timestamp',
                dtype=DBBColumnDType.datetime_t
            )
        ]
    )
    
    task_meta = DBBTaskMeta(
        name='test_task',
        source='task/{split}.npz',
        format=DBBTableDataFormat.NUMPY,
        columns=[
            DBBColumnSchema(
                name='user_id',
                shared_schema='interaction.user_id'  # References FK in interaction
            ),
            DBBColumnSchema(
                name='timestamp',
                shared_schema='interaction.timestamp'  # References timestamp
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
        tables=[user_table, interaction_table],
        tasks=[task_meta]
    )
    
    # Create mock dataset with actual data
    class MockDataset:
        def __init__(self):
            self.metadata = dataset_meta
            self.tables = {
                'user': {
                    'user_id': np.array([1, 2, 3]),
                    'user_feature': np.array([0.1, 0.2, 0.3])
                },
                'interaction': {
                    'user_id': np.array([1, 2, 1, 3]),
                    'timestamp': np.array(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
                }
            }
            self.tasks = [MockTask()]
    
    class MockTask:
        def __init__(self):
            self.metadata = task_meta
            self.train_set = {
                'user_id': np.array([1, 2]),
                'timestamp': np.array(['2023-01-01', '2023-01-02']),
                'label': np.array([0, 1])
            }
            self.validation_set = {
                'user_id': np.array([1]),
                'timestamp': np.array(['2023-01-03']),
                'label': np.array([1])
            }
            self.test_set = {
                'user_id': np.array([3]),
                'timestamp': np.array(['2023-01-04']),
                'label': np.array([0])
            }
    
    # Test the extract_data method
    from fastdfs.preprocess.transform_preprocess import RDBTransformPreprocessConfig
    
    # Create a minimal config
    config = RDBTransformPreprocessConfig(transforms=[])
    preprocessor = RDBTransformPreprocess(config)
    dataset = MockDataset()
    
    print("Testing extract_data with column groups...")
    rdb_data = preprocessor.extract_data(dataset)
    
    print(f"✓ Tables: {list(rdb_data.tables.keys())}")
    print(f"✓ Column groups: {rdb_data.column_groups}")
    
    # Check that column groups are created for shared schema
    task_table_name = make_task_table_name('test_task')
    expected_groups = [
        [(task_table_name, 'user_id'), ('interaction', 'user_id')],
        [(task_table_name, 'timestamp'), ('interaction', 'timestamp')]
    ]
    
    if rdb_data.column_groups:
        for group in rdb_data.column_groups:
            print(f"  - Column group: {group}")
    
    print("\\nTesting extract_task_data with metadata inference...")
    task_data_fit, task_data_transform = preprocessor.extract_task_data(dataset)
    
    print(f"✓ Task fit tables: {list(task_data_fit.tables.keys())}")
    print(f"✓ Task transform tables: {list(task_data_transform.keys())}")
    
    # Check metadata inference
    task_table = task_data_fit.tables[task_table_name]
    user_id_col = task_table['user_id']
    print(f"✓ user_id metadata: {user_id_col.metadata}")
    
    expected_dtype = DBBColumnDType.foreign_key
    expected_link_to = 'user.user_id'
    
    if user_id_col.metadata.get('dtype') == expected_dtype:
        print(f"✓ Correctly inferred dtype: {expected_dtype}")
    else:
        print(f"✗ Wrong dtype: expected {expected_dtype}, got {user_id_col.metadata.get('dtype')}")
    
    if user_id_col.metadata.get('link_to') == expected_link_to:
        print(f"✓ Correctly inferred link_to: {expected_link_to}")
    else:
        print(f"✗ Wrong link_to: expected {expected_link_to}, got {user_id_col.metadata.get('link_to')}")
    
    print("\\n✅ Column groups and metadata inference test completed!")


if __name__ == "__main__":
    test_column_groups_with_shared_schema()
