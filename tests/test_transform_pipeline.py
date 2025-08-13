"""
Unit tests for the complete transform pipeline with shared schema support.
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from pydantic import BaseModel
from fastdfs.dataset.meta import (
    DBBColumnSchema, DBBTaskMeta, DBBTableSchema, DBBRDBDatasetMeta,
    DBBColumnDType, DBBTaskType, DBBTaskEvalMetric, DBBTableDataFormat
)
from fastdfs.preprocess.transform.base import (
    RDBData, ColumnData, ColumnTransform, RDBTransform, make_task_table_name,
    column_transform, rdb_transform
)
from fastdfs.preprocess.transform_preprocess import RDBTransformPreprocess, RDBTransformPreprocessConfig, _merge_rdb_and_task
from fastdfs.utils.device import DeviceInfo


# Custom test transforms with predictable outputs
class NormalizeTransformConfig(BaseModel):
    pass


@column_transform
class NormalizeTransform(ColumnTransform):
    """Simple normalization transform for testing - subtracts mean."""
    name = "test_normalize"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    config_class = NormalizeTransformConfig
    
    def __init__(self, config=None):
        self.mean = None
    
    def fit(self, column: ColumnData, device: DeviceInfo):
        self.mean = np.mean(column.data)
    
    def transform(self, column: ColumnData, device: DeviceInfo):
        if self.mean is None:
            raise ValueError("Transform not fitted")
        
        # Simple mean subtraction for predictable results
        transformed_data = column.data - self.mean
        new_metadata = column.metadata.copy()
        new_metadata['normalized'] = True
        new_metadata['mean'] = self.mean
        
        return [ColumnData(new_metadata, transformed_data)]


class CategoryEncodingTransformConfig(BaseModel):
    pass


@column_transform  
class CategoryEncodingTransform(ColumnTransform):
    """Simple category encoding transform for testing - ordinal encoding."""
    name = "test_category_encode"
    input_dtype = DBBColumnDType.category_t
    output_dtypes = [DBBColumnDType.float_t]
    config_class = CategoryEncodingTransformConfig
    
    def __init__(self, config=None):
        self.categories = None
    
    def fit(self, column: ColumnData, device: DeviceInfo):
        # Sort categories for deterministic encoding
        self.categories = sorted(list(set(column.data)))
    
    def transform(self, column: ColumnData, device: DeviceInfo):
        if self.categories is None:
            raise ValueError("Transform not fitted")
        
        # Handle unseen categories by assigning them the maximum category index + 1
        def encode_value(val):
            try:
                return self.categories.index(val)
            except ValueError:
                # Unseen category - assign it a new index
                return len(self.categories)
        
        # Deterministic ordinal encoding with unseen category handling
        transformed_data = np.array([encode_value(val) for val in column.data], dtype=float)
        new_metadata = column.metadata.copy()
        new_metadata['dtype'] = DBBColumnDType.float_t
        new_metadata['encoded'] = True
        new_metadata['categories'] = self.categories
        
        return [ColumnData(new_metadata, transformed_data)]

@pytest.fixture
def test_dataset():
    """Create a simple test dataset with shared schema."""
    
    # Define table schemas
    user_table = DBBTableSchema(
        name='users',
        source='data/users.npz',
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='user_id',
                dtype=DBBColumnDType.primary_key,
                capacity=100
            ),
            DBBColumnSchema(
                name='age',
                dtype=DBBColumnDType.float_t,
                in_size=1
            ),
            DBBColumnSchema(
                name='category',
                dtype=DBBColumnDType.category_t,
                num_categories=3
            )
        ]
    )
    
    item_table = DBBTableSchema(
        name='items',
        source='data/items.npz', 
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='item_id',
                dtype=DBBColumnDType.primary_key,
                capacity=50
            ),
            DBBColumnSchema(
                name='price',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ]
    )
    
    interaction_table = DBBTableSchema(
        name='interactions',
        source='data/interactions.npz',
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='user_id',
                dtype=DBBColumnDType.foreign_key,
                link_to='users.user_id',
                capacity=100
            ),
            DBBColumnSchema(
                name='item_id', 
                dtype=DBBColumnDType.foreign_key,
                link_to='items.item_id',
                capacity=50
            ),
            DBBColumnSchema(
                name='rating',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ]
    )
    
    # Define task with shared schema
    task_meta = DBBTaskMeta(
        name='rating_prediction',
        source='task/{split}.npz',
        format=DBBTableDataFormat.NUMPY,
        columns=[
            DBBColumnSchema(
                name='user_id',
                shared_schema='interactions.user_id'
            ),
            DBBColumnSchema(
                name='item_id',
                shared_schema='interactions.item_id'
            ),
            DBBColumnSchema(
                name='user_age',
                shared_schema='users.age'
            ),
            DBBColumnSchema(
                name='target_rating',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ],
        evaluation_metric=DBBTaskEvalMetric.mse,
        target_column='target_rating',
        task_type=DBBTaskType.regression
    )
    
    dataset_meta = DBBRDBDatasetMeta(
        dataset_name='test_rating_dataset',
        tables=[user_table, item_table, interaction_table],
        tasks=[task_meta]
    )
    
    # Create mock dataset with predictable data
    class MockDataset:
        def __init__(self):
            self.metadata = dataset_meta
            self.tables = {
                'users': {
                    'user_id': np.array([1, 2, 3, 4, 5]),
                    'age': np.array([20.0, 30.0, 40.0, 50.0, 60.0]),  # Nice round numbers
                    'category': np.array(['A', 'B', 'C', 'A', 'B'])
                },
                'items': {
                    'item_id': np.array([10, 11, 12]),
                    'price': np.array([10.0, 20.0, 30.0])  # Simple values
                },
                'interactions': {
                    'user_id': np.array([1, 2, 3, 1, 2]),
                    'item_id': np.array([10, 11, 12, 11, 10]),
                    'rating': np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Sequential values
                }
            }
            self.tasks = [MockTask()]
    
    class MockTask:
        def __init__(self):
            self.metadata = task_meta
            # Training data
            self.train_set = {
                'user_id': np.array([1, 2, 3]),
                'item_id': np.array([10, 11, 12]),
                'user_age': np.array([20.0, 30.0, 40.0]),
                'target_rating': np.array([1.0, 2.0, 3.0])
            }
            # Validation data
            self.validation_set = {
                'user_id': np.array([1, 2]),
                'item_id': np.array([11, 10]),
                'user_age': np.array([20.0, 30.0]),
                'target_rating': np.array([4.0, 5.0])
            }
            # Test data
            self.test_set = {
                'user_id': np.array([4, 5]),
                'item_id': np.array([10, 11]),
                'user_age': np.array([50.0, 60.0]),
                'target_rating': np.array([2.5, 3.5])
            }
    
    return MockDataset()


@pytest.fixture
def device():
    """Create a test device info."""
    return DeviceInfo(cpu_count=1, gpu_devices=[])


def test_data_extraction(test_dataset, device):
    """Test data extraction with shared schema support."""
    # Create simple config with no transforms for basic testing
    config = RDBTransformPreprocessConfig(transforms=[])
    preprocessor = RDBTransformPreprocess(config)
    
    # Test extract_data
    rdb_data = preprocessor.extract_data(test_dataset)
    assert len(rdb_data.tables) == 3
    assert 'users' in rdb_data.tables
    assert 'items' in rdb_data.tables  
    assert 'interactions' in rdb_data.tables
    
    # Test column groups creation
    assert rdb_data.column_groups is not None
    assert len(rdb_data.column_groups) == 3  # user_id, item_id, user_age groups
    
    # Test extract_task_data
    task_data_fit, task_data_transform = preprocessor.extract_task_data(test_dataset)
    assert len(task_data_fit.tables) == 1
    assert len(task_data_transform) == 1


def test_shared_schema_inference(test_dataset, device):
    """Test that shared schema metadata inference works correctly."""
    config = RDBTransformPreprocessConfig(transforms=[])
    preprocessor = RDBTransformPreprocess(config)
    
    task_data_fit, _ = preprocessor.extract_task_data(test_dataset)
    task_table_name = make_task_table_name('rating_prediction')
    assert task_table_name in task_data_fit.tables
    
    task_table = task_data_fit.tables[task_table_name]
    
    # Check shared schema references
    assert task_table['user_id'].metadata['shared_schema'] == 'interactions.user_id'
    assert task_table['item_id'].metadata['shared_schema'] == 'interactions.item_id'
    assert task_table['user_age'].metadata['shared_schema'] == 'users.age'
    
    # Check inferred dtypes
    assert task_table['user_id'].metadata['dtype'] == DBBColumnDType.foreign_key
    assert task_table['item_id'].metadata['dtype'] == DBBColumnDType.foreign_key
    assert task_table['user_age'].metadata['dtype'] == DBBColumnDType.float_t
    assert task_table['target_rating'].metadata['dtype'] == DBBColumnDType.float_t


def test_column_groups_creation(test_dataset, device):
    """Test that column groups are created correctly for shared schema relationships."""
    config = RDBTransformPreprocessConfig(transforms=[])
    preprocessor = RDBTransformPreprocess(config)
    
    rdb_data = preprocessor.extract_data(test_dataset)
    
    # Check that we have the expected column groups
    assert len(rdb_data.column_groups) == 3
    
    # Convert to sets for easier comparison
    groups = [set(group) for group in rdb_data.column_groups]
    
    # Check for user_id group
    user_id_group = {('__task__:rating_prediction', 'user_id'), ('interactions', 'user_id')}
    assert user_id_group in groups
    
    # Check for item_id group  
    item_id_group = {('__task__:rating_prediction', 'item_id'), ('interactions', 'item_id')}
    assert item_id_group in groups
    
    # Check for user_age group
    user_age_group = {('__task__:rating_prediction', 'user_age'), ('users', 'age')}
    assert user_age_group in groups


def test_transform_pipeline_with_custom_transforms(test_dataset, device):
    """Test the complete transform pipeline with custom transforms and verify values."""
    # Create config with only column transforms (no RDB transforms to avoid complexity)
    config = RDBTransformPreprocessConfig(
        transforms=[
            {"name": "column_transform_chain", "config": {
                "transforms": [
                    {"name": "test_normalize"},
                    {"name": "test_category_encode"}
                ]
            }}
        ]
    )
    
    preprocessor = RDBTransformPreprocess(config)
    
    # Extract data
    rdb_data = preprocessor.extract_data(test_dataset)
    task_data_fit, task_data_transform = preprocessor.extract_task_data(test_dataset)
    
    # Store original values for comparison
    original_age_rdb = rdb_data.tables['users']['age'].data.copy()
    original_category_rdb = rdb_data.tables['users']['category'].data.copy()
    
    # Get task data values
    task_table_name = '__task__:rating_prediction'
    original_age_task_train = task_data_fit.tables[task_table_name]['user_age'].data.copy()
    
    # Fit transforms on combined data (RDB + task training data)
    all_data_fit = _merge_rdb_and_task(rdb_data, task_data_fit)
    for transform in preprocessor.transforms:
        transform.fit(all_data_fit, device)
    
    # Apply transforms to RDB data
    transformed_rdb = rdb_data
    for transform in preprocessor.transforms:
        transformed_rdb = transform.transform(transformed_rdb, device)
    
    # Apply transforms to task data (validation/test)
    transformed_task_data = {}
    for task_name, task_rdb in task_data_transform.items():
        transformed_task = task_rdb
        for transform in preprocessor.transforms:
            transformed_task = transform.transform(transformed_task, device)
        transformed_task_data[task_name] = transformed_task
    
    # Verify transform results
    
    # 1. Check TestNormalizeTransform on age data
    # The normalize transform should subtract the mean calculated from ALL data (RDB + training)
    
    # Combined age data for mean calculation:
    # RDB users.age: [20, 30, 40, 50, 60] 
    # Task training user_age: [20, 30, 40] (from shared_schema users.age)
    # All age values: [20, 30, 40, 50, 60, 20, 30, 40] = mean = 36.25
    all_age_values = np.concatenate([original_age_rdb, original_age_task_train])
    expected_mean = np.mean(all_age_values)
    print(f"Expected mean from combined data: {expected_mean}")
    
    # Check RDB data transformation
    expected_rdb_age = original_age_rdb - expected_mean
    actual_rdb_age = transformed_rdb.tables['users']['age'].data
    print(f"RDB age - Original: {original_age_rdb}, Expected: {expected_rdb_age}, Actual: {actual_rdb_age}")
    
    assert np.allclose(actual_rdb_age, expected_rdb_age), f"RDB age mismatch: expected {expected_rdb_age}, got {actual_rdb_age}"
    assert transformed_rdb.tables['users']['age'].metadata['normalized'] == True
    assert abs(transformed_rdb.tables['users']['age'].metadata['mean'] - expected_mean) < 1e-6
    
    # Check task data transformation (validation/test should use fitted transform)
    task_data = transformed_task_data['rating_prediction']
    actual_task_age = task_data.tables[task_table_name]['user_age'].data
    
    # Task validation user_age: [20, 30] -> normalized: [20-36.25, 30-36.25] = [-16.25, -6.25]
    # Task test user_age: [50, 60] -> normalized: [50-36.25, 60-36.25] = [13.75, 23.75]
    validation_age = np.array([20.0, 30.0])  # From test_dataset fixture
    test_age = np.array([50.0, 60.0])  # From test_dataset fixture
    expected_validation_age = validation_age - expected_mean
    expected_test_age = test_age - expected_mean
    expected_task_age = np.concatenate([expected_validation_age, expected_test_age])
    
    print(f"Task age - Expected: {expected_task_age}, Actual: {actual_task_age}")
    assert np.allclose(actual_task_age, expected_task_age), f"Task age mismatch: expected {expected_task_age}, got {actual_task_age}"
    
    # 2. Check TestCategoryEncodingTransform with new categories in validation/test
    # First, let's update the test dataset to have new categories in validation/test
    
    # RDB categories: ['A', 'B', 'C', 'A', 'B'] -> unique sorted: ['A', 'B', 'C'] -> encoding: [0, 1, 2, 0, 1]
    # The transform should fit on RDB + training data
    all_category_values = np.concatenate([
        original_category_rdb,  # ['A', 'B', 'C', 'A', 'B']
        # Training categories are determined by shared_schema, but let's check what's actually in the task
    ])
    
    # Check category encoding on RDB data
    expected_categories = ['A', 'B', 'C']  # Sorted unique categories
    expected_rdb_category = np.array([0.0, 1.0, 2.0, 0.0, 1.0])  # [A, B, C, A, B] -> [0, 1, 2, 0, 1]
    actual_rdb_category = transformed_rdb.tables['users']['category'].data
    
    print(f"RDB category - Original: {original_category_rdb}, Expected: {expected_rdb_category}, Actual: {actual_rdb_category}")
    assert np.allclose(actual_rdb_category, expected_rdb_category), f"RDB category mismatch: expected {expected_rdb_category}, got {actual_rdb_category}"
    assert transformed_rdb.tables['users']['category'].metadata['encoded'] == True
    assert transformed_rdb.tables['users']['category'].metadata['categories'] == expected_categories
    
    print("✅ All transform value checks passed!")


# Update the test dataset to include new categories in validation/test
@pytest.fixture
def test_dataset_with_new_categories():
    """Create a test dataset with new categories in validation/test sets."""
    
    # Define table schemas (same as before)
    user_table = DBBTableSchema(
        name='users',
        source='data/users.npz',
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='user_id',
                dtype=DBBColumnDType.primary_key,
                capacity=100
            ),
            DBBColumnSchema(
                name='age',
                dtype=DBBColumnDType.float_t,
                in_size=1
            ),
            DBBColumnSchema(
                name='category',
                dtype=DBBColumnDType.category_t,
                num_categories=5  # Increased to handle new categories
            )
        ]
    )
    
    item_table = DBBTableSchema(
        name='items',
        source='data/items.npz', 
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='item_id',
                dtype=DBBColumnDType.primary_key,
                capacity=50
            ),
            DBBColumnSchema(
                name='price',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ]
    )
    
    interaction_table = DBBTableSchema(
        name='interactions',
        source='data/interactions.npz',
        format=DBBTableDataFormat.NUMPY,
        time_column=None,
        columns=[
            DBBColumnSchema(
                name='user_id',
                dtype=DBBColumnDType.foreign_key,
                link_to='users.user_id',
                capacity=100
            ),
            DBBColumnSchema(
                name='item_id', 
                dtype=DBBColumnDType.foreign_key,
                link_to='items.item_id',
                capacity=50
            ),
            DBBColumnSchema(
                name='rating',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ]
    )
    
    # Define task with shared schema including category
    task_meta = DBBTaskMeta(
        name='rating_prediction',
        source='task/{split}.npz',
        format=DBBTableDataFormat.NUMPY,
        columns=[
            DBBColumnSchema(
                name='user_id',
                shared_schema='interactions.user_id'
            ),
            DBBColumnSchema(
                name='item_id',
                shared_schema='interactions.item_id'
            ),
            DBBColumnSchema(
                name='user_age',
                shared_schema='users.age'
            ),
            DBBColumnSchema(
                name='user_category',  # Add category to task
                shared_schema='users.category'
            ),
            DBBColumnSchema(
                name='target_rating',
                dtype=DBBColumnDType.float_t,
                in_size=1
            )
        ],
        evaluation_metric=DBBTaskEvalMetric.mse,
        target_column='target_rating',
        task_type=DBBTaskType.regression
    )
    
    dataset_meta = DBBRDBDatasetMeta(
        dataset_name='test_rating_dataset_with_new_categories',
        tables=[user_table, item_table, interaction_table],
        tasks=[task_meta]
    )
    
    # Create mock dataset with new categories in validation/test
    class MockDataset:
        def __init__(self):
            self.metadata = dataset_meta
            self.tables = {
                'users': {
                    'user_id': np.array([1, 2, 3, 4, 5]),
                    'age': np.array([20.0, 30.0, 40.0, 50.0, 60.0]),
                    'category': np.array(['A', 'B', 'C', 'A', 'B'])  # Only A, B, C in training
                },
                'items': {
                    'item_id': np.array([10, 11, 12]),
                    'price': np.array([10.0, 20.0, 30.0])
                },
                'interactions': {
                    'user_id': np.array([1, 2, 3, 1, 2]),
                    'item_id': np.array([10, 11, 12, 11, 10]),
                    'rating': np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                }
            }
            self.tasks = [MockTask()]
    
    class MockTask:
        def __init__(self):
            self.metadata = task_meta
            # Training data - only known categories A, B, C
            self.train_set = {
                'user_id': np.array([1, 2, 3]),
                'item_id': np.array([10, 11, 12]),
                'user_age': np.array([20.0, 30.0, 40.0]),
                'user_category': np.array(['A', 'B', 'C']),  # Known categories
                'target_rating': np.array([1.0, 2.0, 3.0])
            }
            # Validation data - includes new category D
            self.validation_set = {
                'user_id': np.array([1, 2]),
                'item_id': np.array([11, 10]),
                'user_age': np.array([20.0, 30.0]),
                'user_category': np.array(['A', 'D']),  # D is new!
                'target_rating': np.array([4.0, 5.0])
            }
            # Test data - includes new category E  
            self.test_set = {
                'user_id': np.array([4, 5]),
                'item_id': np.array([10, 11]),
                'user_age': np.array([50.0, 60.0]),
                'user_category': np.array(['E', 'B']),  # E is new!
                'target_rating': np.array([2.5, 3.5])
            }
    
    return MockDataset()


def test_transform_with_new_categories(test_dataset_with_new_categories, device):
    """Test transform pipeline with new/unseen categories in validation/test sets."""
    config = RDBTransformPreprocessConfig(
        transforms=[
            {"name": "column_transform_chain", "config": {
                "transforms": [
                    {"name": "test_normalize"},
                    {"name": "test_category_encode"}
                ]
            }}
        ]
    )
    
    preprocessor = RDBTransformPreprocess(config)
    
    # Extract data
    rdb_data = preprocessor.extract_data(test_dataset_with_new_categories)
    task_data_fit, task_data_transform = preprocessor.extract_task_data(test_dataset_with_new_categories)
    
    # Fit transforms on combined data (RDB + task training data)
    all_data_fit = _merge_rdb_and_task(rdb_data, task_data_fit)
    for transform in preprocessor.transforms:
        transform.fit(all_data_fit, device)
    
    # Apply transforms to RDB data
    transformed_rdb = rdb_data
    for transform in preprocessor.transforms:
        transformed_rdb = transform.transform(transformed_rdb, device)
    
    # Apply transforms to task data (validation/test with unseen categories)
    transformed_task_data = {}
    for task_name, task_rdb in task_data_transform.items():
        transformed_task = task_rdb
        for transform in preprocessor.transforms:
            transformed_task = transform.transform(transformed_task, device)
        transformed_task_data[task_name] = transformed_task
    
    # Verify transform results
    
    # 1. Check that category encoding was fitted correctly on RDB + training data
    # Combined categories: RDB['A', 'B', 'C', 'A', 'B'] + Training['A', 'B', 'C'] = unique sorted ['A', 'B', 'C']
    expected_categories = ['A', 'B', 'C']
    actual_categories = transformed_rdb.tables['users']['category'].metadata['categories']
    assert actual_categories == expected_categories, f"Expected categories {expected_categories}, got {actual_categories}"
    
    # 2. Check RDB category encoding
    expected_rdb_encoded = np.array([0.0, 1.0, 2.0, 0.0, 1.0])  # ['A', 'B', 'C', 'A', 'B'] -> [0, 1, 2, 0, 1]
    actual_rdb_encoded = transformed_rdb.tables['users']['category'].data
    assert np.allclose(actual_rdb_encoded, expected_rdb_encoded), f"RDB encoding: expected {expected_rdb_encoded}, got {actual_rdb_encoded}"
    
    # 3. Check task data category encoding with unseen categories
    task_table_name = '__task__:rating_prediction'
    task_data = transformed_task_data['rating_prediction']
    task_encoded = task_data.tables[task_table_name]['user_category'].data
    
    # Expected task encoding:
    # Validation: ['A', 'D'] -> [0, 3] (A is known=0, D is unseen gets index 3)
    # Test: ['E', 'B'] -> [3, 1] (E is unseen gets index 3, B is known=1)
    # Combined: [0, 3, 3, 1] (all unseen categories get the same index len(categories))
    expected_task_encoded = np.array([0.0, 3.0, 3.0, 1.0])  # A=0, D=3 (unseen), E=3 (unseen), B=1
    
    print(f"Task category encoding - Expected: {expected_task_encoded}, Actual: {task_encoded}")
    assert np.allclose(task_encoded, expected_task_encoded), f"Task encoding: expected {expected_task_encoded}, got {task_encoded}"
    
    # 4. Check age normalization across all data
    # All age data for fitting: RDB [20,30,40,50,60] + Training [20,30,40] = [20,30,40,50,60,20,30,40]
    # Mean = (20+30+40+50+60+20+30+40)/8 = 290/8 = 36.25
    
    rdb_age = rdb_data.tables['users']['age'].data  # [20,30,40,50,60]
    task_train_age = task_data_fit.tables[task_table_name]['user_age'].data  # [20,30,40]
    all_age_for_fitting = np.concatenate([rdb_age, task_train_age])
    expected_mean = np.mean(all_age_for_fitting)
    
    # Check RDB age normalization
    expected_rdb_age_normalized = rdb_age - expected_mean
    actual_rdb_age_normalized = transformed_rdb.tables['users']['age'].data
    assert np.allclose(actual_rdb_age_normalized, expected_rdb_age_normalized), f"RDB age normalization failed"
    
    # Check task age normalization
    # Task validation+test ages: [20, 30, 50, 60] -> normalized: [20-36.25, 30-36.25, 50-36.25, 60-36.25]
    task_val_test_ages = np.array([20.0, 30.0, 50.0, 60.0])  # From validation and test sets
    expected_task_age_normalized = task_val_test_ages - expected_mean
    actual_task_age_normalized = task_data.tables[task_table_name]['user_age'].data
    
    print(f"Task age normalization - Expected: {expected_task_age_normalized}, Actual: {actual_task_age_normalized}")
    assert np.allclose(actual_task_age_normalized, expected_task_age_normalized), f"Task age normalization failed"
    
    print("✅ Transform with new categories test passed!")


def test_column_groups_functionality(test_dataset, device):
    """Test that column groups enable proper transform coordination."""
    config = RDBTransformPreprocessConfig(transforms=[])
    preprocessor = RDBTransformPreprocess(config)
    
    # Extract data and check column groups
    rdb_data = preprocessor.extract_data(test_dataset)
    task_data_fit, _ = preprocessor.extract_task_data(test_dataset)
    
    # Verify column groups in RDB data
    assert rdb_data.column_groups is not None
    assert len(rdb_data.column_groups) == 3
    
    # Verify column groups in task data
    assert task_data_fit.column_groups is not None
    assert len(task_data_fit.column_groups) == 3
    
    # Check that groups contain expected relationships
    all_groups = rdb_data.column_groups + task_data_fit.column_groups
    
    # Find groups by their content
    user_id_groups = [g for g in all_groups if any('user_id' in str(item) for item in g)]
    item_id_groups = [g for g in all_groups if any('item_id' in str(item) for item in g)]
    age_groups = [g for g in all_groups if any('age' in str(item) for item in g)]
    
    assert len(user_id_groups) > 0
    assert len(item_id_groups) > 0  
    assert len(age_groups) > 0


if __name__ == "__main__":
    # For backwards compatibility, allow running as script
    import pytest
    pytest.main([__file__, "-v"])
