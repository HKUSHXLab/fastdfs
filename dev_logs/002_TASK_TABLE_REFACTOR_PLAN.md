# Task Table Refactor Plan

## Overview
The goal is to make task table definition more flexible by removing the requirement for task tables to mirror the schema of a corresponding data table in the RDB (as specified by the target_table field).

## Current Design Analysis

### Current Task Table Design
1. **target_table field**: Task tables currently require a `target_table` field that specifies which data table schema they should mirror
2. **Schema mirroring**: Task table columns must match the schema of the target table
3. **Key fields**: Tasks have `key_prediction_label_column` and `key_prediction_query_idx_column` fields specifically for retrieval tasks
4. **Task table naming**: Task tables are named using `make_task_table_name(task_name, target_table_name)` format

### Current Code Structure
1. **Metadata definitions**: In `fastdfs/dataset/meta.py` - `DBBTaskMeta` class defines task structure
2. **Transform preprocessing**: In `fastdfs/preprocess/transform_preprocess.py` - handles task data extraction and processing
3. **DFS preprocessing**: In `fastdfs/preprocess/dfs/core.py` - uses target_table for relationship building
4. **Transform modules**: Various transform classes use target_table for schema understanding

## Proposed New Design

### 1. New Column Schema Field
- Add an optional `shared_schema` field to `DBBColumnSchema` class
- Format: `"table_name.column_name"` (string)
- When specified, the column shares schema with the referenced data table column
- When not specified, the column is task-specific

### 2. Simplified Task Metadata
- Remove `target_table` field from `DBBTaskMeta`
- Remove `key_prediction_label_column` and `key_prediction_query_idx_column` fields
- Keep other essential fields like `name`, `target_column`, `evaluation_metric`, etc.

### 3. Updated Task Table Naming
- Modify `make_task_table_name()` to not require target_table_name
- Use format like `__task__:{task_name}` instead of `__task__:{task_name}:{target_table_name}`

## Detailed Work Plan

### Phase 1: Update Metadata Schema (Core Changes)
1. **Update `fastdfs/dataset/meta.py`**:
   - Add `shared_schema: Optional[str] = None` field to `DBBColumnSchema`
   - Remove `target_table` field from `DBBTaskMeta`
   - Remove `key_prediction_label_column` and `key_prediction_query_idx_column` fields
   - Update `TASK_EXTRA_FIELDS` to remove retrieval-specific fields

2. **Update `fastdfs/preprocess/transform/base.py`**:
   - Modify `make_task_table_name()` to only take task_name parameter
   - Update `unmake_task_table_name()` to handle new naming format
   - Update any related helper functions

3. **Update test metadata `tests/data/test_rdb/metadata.yaml`**:
   - Add `shared_schema` fields to task columns where appropriate
   - Remove `target_table`, `key_prediction_label_column`, `key_prediction_query_idx_column` fields
   - Update column definitions to use new shared_schema format

### Phase 2: Update Transform Preprocessing
1. **Update `fastdfs/preprocess/transform_preprocess.py`**:
   - Modify `extract_task_data()` to not rely on target_table
   - Update task table name generation
   - Change logic for determining which columns need fit vs transform
   - Update `output_tasks()` to handle new schema
   - Remove retrieval-specific column handling

2. **Update transform modules in `fastdfs/preprocess/transform/`**:
   - **`key_mapping.py`**: Update to use shared_schema instead of target_table for understanding relationships
   - **`dummy_table.py`**: Update any target_table dependencies
   - **Other transform modules**: Update any code that relies on target_table

### Phase 3: Update DFS Preprocessing
1. **Update `fastdfs/preprocess/dfs/core.py`**:
   - Modify `build_dataframes()` to not rely on target_table
   - Update relationship building logic to use shared_schema information
   - Change foreign key detection logic to use shared_schema
   - Update task dataframe construction

2. **Update `fastdfs/preprocess/dfs/dfs_preprocess.py`**:
   - Modify `_infer_dtype()` to handle new schema approach
   - Update any target_table usage

3. **Keep `fastdfs/preprocess/dfs/gen_sqls.py` unchanged**:
   - As specified, the target_table concept in gen_sqls.py is different and should not be modified

### Phase 4: Update Data Loading/Saving
1. **Update `fastdfs/dataset/rdb_dataset.py`**:
   - Modify `DBBRDBTaskCreator` class
   - Remove `set_target_table()`, `set_key_prediction_label_column()`, `set_key_prediction_query_idx_column()` methods
   - Update task creation logic

2. **Update dataset loading/saving logic**:
   - Ensure new metadata format is properly loaded and saved
   - Update validation logic

### Phase 5: Add Unit Tests
1. **Create comprehensive unit tests in `tests/`**:
   - Test data loading with new schema format
   - Test data saving with new schema format
   - Test transform preprocessing with shared_schema
   - Test DFS preprocessing with new design
   - Test relationship inference from shared_schema
   - Test backward compatibility (if needed)

2. **Test files to create**:
   - `tests/test_task_metadata_refactor.py`
   - `tests/test_shared_schema_loading.py`
   - `tests/test_transform_with_shared_schema.py`
   - `tests/test_dfs_with_shared_schema.py`

### Phase 6: Documentation and Cleanup
1. **Update documentation**:
   - Update examples to use new schema format
   - Update README if necessary
   - Add migration guide for existing datasets

2. **Clean up unused code**:
   - Remove any unused target_table references
   - Remove retrieval-specific handling code
   - Clean up import statements

## Key Implementation Considerations

### Relationship Inference from shared_schema
- Parse `shared_schema` field format: `"table_name.column_name"`
- Use this to determine if a task column is a primary key or foreign key
- Build relationships based on the referenced table schema
- Handle cases where shared_schema is not specified (task-specific columns)

### Backward Compatibility
- Consider whether to maintain backward compatibility with old metadata format
- If yes, add migration logic to convert old format to new format
- If no, ensure clear error messages for old format

### Error Handling
- Add validation for shared_schema format
- Ensure referenced table.column exists in the dataset
- Handle missing or invalid shared_schema references gracefully

### Performance Considerations
- Ensure the new design doesn't significantly impact performance
- Consider caching parsed shared_schema information
- Optimize relationship building logic

## Dependencies and Risks

### Dependencies
- Changes to metadata schema affect all downstream processing
- Transform and DFS modules have tight coupling with current design
- Test data must be updated to reflect new format

### Risks
- Breaking changes to existing datasets and workflows
- Complex refactoring across multiple modules
- Potential for introducing bugs in relationship inference
- Need thorough testing to ensure feature parity

### Mitigation Strategies
- Implement changes incrementally with thorough testing at each phase
- Maintain comprehensive unit tests
- Consider feature flags for gradual rollout
- Document all changes clearly for users

## Success Criteria
1. Task tables no longer require target_table field
2. Columns can optionally specify shared_schema for schema sharing
3. All existing functionality works with new design
4. Comprehensive unit tests pass
5. Example datasets work with new format
6. Performance is maintained or improved
