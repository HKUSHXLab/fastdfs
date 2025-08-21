# Task Table Refactor Implementation Update

## Overview
Updated the task table refactor implementation based on feedback to properly handle shared schema, column groups, and metadata inference according to the FastDFS architecture.

## Key Changes Made

### 1. Fixed Fit-Transform Logic ✅
**Issue**: Transform decisions were incorrectly based on `shared_schema` rather than train vs validation/test splits.

**Solution**: Updated `extract_task_data()` in `transform_preprocess.py`:
- Training data always goes to `fit_table` for learning transform parameters
- Validation and test data go to `transform_tables` for applying learned transforms
- This follows the correct pattern regardless of whether columns have `shared_schema`

```python
# Training data for fitting transforms
fit_table[task_table_name][col] = ColumnData(col_meta, train_data)

# Val/test data for applying transforms
val_test_data = np.concatenate([task.validation_set[col], task.test_set[col]], axis=0)
transform_tables[task_name][task_table_name][col] = ColumnData(col_meta, val_test_data)
```

### 2. Column Group Creation Based on shared_schema ✅
**Issue**: Column groups weren't being created to leverage shared schema for transform processing.

**Solution**: Enhanced `extract_data()` and `extract_task_data()` to create column groups:
- Columns with the same `shared_schema` are grouped together
- Both task columns and referenced data columns are included in the same group
- This enables transform routines to process related columns together

```python
# Create column groups for shared schema
shared_schema_groups = {}
for col_schema in task.metadata.columns:
    if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
        shared_schema = col_schema.shared_schema
        # Group task column with its referenced data column
        task_col_ref = (task_table_name, col_schema.name)
        data_table, data_col = shared_schema.split('.')
        data_col_ref = (data_table, data_col)
        
        if shared_schema not in shared_schema_groups:
            shared_schema_groups[shared_schema] = []
        shared_schema_groups[shared_schema].extend([task_col_ref, data_col_ref])
```

### 3. Automatic Metadata Inference ✅
**Issue**: Redundant metadata fields (dtype, link_to, capacity) needed to be specified even when shared_schema was provided.

**Solution**: 
1. Made `dtype` optional in schema when `shared_schema` is provided
2. Added automatic metadata inference from referenced data columns

```python
# Schema validation - dtype optional when shared_schema provided
@pydantic.root_validator
def validate_dtype_or_shared_schema(cls, values):
    dtype = values.get('dtype')
    shared_schema = values.get('shared_schema')
    if dtype is None and not shared_schema:
        raise ValueError('dtype is required when shared_schema is not provided')
    return values

# Metadata inference during data loading
if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
    shared_schema_key = col_schema.shared_schema
    if shared_schema_key in data_schema_map:
        ref_schema = data_schema_map[shared_schema_key]
        for field in ['dtype', 'link_to', 'capacity']:
            if field not in col_meta or col_meta[field] is None:
                if hasattr(ref_schema, field):
                    col_meta[field] = getattr(ref_schema, field)
```

### 4. Simplified Test Metadata ✅
**Issue**: Test metadata contained redundant fields that could be inferred.

**Solution**: Updated `tests/data/test_rdb/metadata.yaml`:
```yaml
# Before - redundant fields
- capacity: 100
  dtype: foreign_key
  link_to: user.user_id
  name: user_id
  shared_schema: interaction.user_id

# After - clean and inferred
- name: user_id
  shared_schema: interaction.user_id
```

## Architecture Alignment

### Column Groups for Transform Processing
The implementation now properly leverages the FastDFS transform architecture:

1. **RDBData Structure**: Column groups are populated based on `shared_schema` relationships
2. **Transform Reuse**: Existing transform logic can process grouped columns together
3. **Wrapper Pattern**: The `RDBTransformWrapper` can apply column transforms to entire groups
4. **Metadata Consistency**: All columns in a group share the same logical metadata

### Fit-Transform Separation
Follows the correct scikit-learn pattern:
- **Fit Phase**: Learn parameters from training data only
- **Transform Phase**: Apply learned parameters to validation/test data
- **Shared Processing**: Columns with `shared_schema` benefit from shared transform parameters

## Test Results

### Column Groups Creation ✅
```
✓ Tables: ['user', 'interaction']
✓ Column groups: [
    [('__task__:test_task', 'user_id'), ('interaction', 'user_id')],
    [('__task__:test_task', 'timestamp'), ('interaction', 'timestamp')]
  ]
```

### Metadata Inference ✅
```
✓ user_id metadata: {
    'name': 'user_id', 
    'dtype': 'foreign_key', 
    'shared_schema': 'interaction.user_id', 
    'link_to': 'user.user_id', 
    'capacity': 100
  }
✓ Correctly inferred dtype: foreign_key
✓ Correctly inferred link_to: user.user_id
```

### End-to-End Integration ✅
```
✅ All tests passed! Task table refactor is working correctly.
```

## Benefits Achieved

### 1. Clean Metadata
- No redundant field specification required
- Automatic inference from referenced data columns
- Validation ensures either `dtype` or `shared_schema` is provided

### 2. Efficient Processing
- Column groups enable batch processing of related columns
- Transform routines can leverage shared parameters
- Consistent metadata across grouped columns

### 3. Architectural Compliance
- Proper fit-transform separation
- Leverage existing transform infrastructure
- Column group pattern enables transform reuse

### 4. Developer Experience
- Simplified metadata specification
- Clear shared schema relationships
- Automatic validation and inference

## Implementation Status

✅ **Core Metadata Schema**: Updated with optional dtype and shared_schema support
✅ **Transform Preprocessing**: Fixed fit-transform logic and added column group creation  
✅ **Metadata Inference**: Automatic inference from referenced data columns
✅ **Test Data**: Simplified format with inferred fields
✅ **Integration Testing**: All tests passing with new architecture
✅ **Column Groups**: Proper creation and utilization for shared schema
✅ **Architecture Alignment**: Follows FastDFS design patterns

The refactor now properly implements the intended architecture where:
- Task columns can flexibly reference data table schemas via `shared_schema`
- Transform processing leverages column groups for efficiency
- Metadata is automatically inferred to reduce redundancy
- The implementation aligns with the established FastDFS patterns

This provides a clean, efficient, and maintainable foundation for flexible task table definitions while maximizing reuse of existing transform infrastructure.
