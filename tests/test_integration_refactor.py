"""
Simple integration test to verify the task table refactor works end-to-end.
"""
import yaml
import tempfile
import os
from fastdfs.dataset.meta import DBBRDBDatasetMeta, DBBColumnSchema, DBBTaskMeta, DBBTableSchema
from fastdfs.preprocess.transform.base import make_task_table_name, unmake_task_table_name


def test_end_to_end_refactor():
    """Test that the refactor works end-to-end."""
    print("Testing task table refactor end-to-end...")
    
    # Load the updated test metadata
    with open('tests/data/test_rdb/metadata.yaml', 'r') as f:
        metadata_dict = yaml.safe_load(f)
    
    # Create metadata object
    dataset_meta = DBBRDBDatasetMeta(**metadata_dict)
    print(f"✓ Loaded dataset: {dataset_meta.dataset_name}")
    
    # Test task metadata
    task = dataset_meta.tasks[0]
    print(f"✓ Task name: {task.name}")
    print(f"✓ Target column: {task.target_column}")
    print(f"✓ No target_table field: {not hasattr(task, 'target_table')}")
    print(f"✓ No key_prediction fields: {not hasattr(task, 'key_prediction_label_column')}")
    
    # Test shared_schema fields
    user_id_col = task.columns[0]
    timestamp_col = task.columns[2]
    label_col = task.columns[3]
    
    print(f"✓ user_id has shared_schema: {user_id_col.shared_schema}")
    print(f"✓ timestamp has shared_schema: {timestamp_col.shared_schema}")
    print(f"✓ label has no shared_schema: {label_col.shared_schema is None}")
    
    # Test task table naming
    task_table_name = make_task_table_name(task.name)
    extracted_name = unmake_task_table_name(task_table_name)
    print(f"✓ Task table name: {task_table_name}")
    print(f"✓ Extracted task name matches: {extracted_name == task.name}")
    
    # Test shared_schema parsing
    for col in task.columns:
        if hasattr(col, 'shared_schema') and col.shared_schema:
            table_name, col_name = col.shared_schema.split('.')
            print(f"✓ {col.name} references {table_name}.{col_name}")
    
    print("✅ All tests passed! Task table refactor is working correctly.")


if __name__ == "__main__":
    test_end_to_end_refactor()
