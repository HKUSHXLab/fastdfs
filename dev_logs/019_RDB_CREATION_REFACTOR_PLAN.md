# Dev Plan: RDB Creation Refactor & API Consolidation

## 1. Goal
Unify the instantiation logic of the `RDB` class by refactoring existing "wither" methods (`create_new_...`) to use the standard `__init__` constructor. Additionally, consolidate the mutation API into a consistent set of `add_` and `update_` methods to support both single-item and bulk operations.

## 2. Analysis of Current State

Currently, `RDB` supports two distinct creation patterns:

1.  **Standard Constructor (`__init__`)**:
    -   Handles disk loading (if `path` provided) or in-memory initialization (if `tables`/`metadata` provided).
    -   Correctly sets `path=None` for in-memory instances.

2.  **Transformation Helpers (`create_new_with_tables_and_metadata`)**:
    -   Used by Transforms and `add_table` to produce modified immutable copies.
    -   Currently uses `__new__` to bypass initialization and manually copies `self.path` (which is semantically incorrect for modified datasets).
    -   Acts as a "Bulk Upsert" primitive.

## 3. Proposal: Unified Mutation API

We will refactor the internal implementation to use `__init__` and expose a clean, consistent public API.

### 3.1 Core Primitive: `update_tables` (Bulk Upsert)
This will replace `create_new_with_tables_and_metadata` as the foundation.

```python
def update_tables(
    self, 
    tables: Optional[Dict[str, pd.DataFrame]] = None, 
    metadata: Optional[Dict[str, RDBTableSchema]] = None
) -> 'RDB':
    """
    Create a new RDB with updated/added tables and metadata.
    This acts as a 'Bulk Upsert' - existing keys are updated, new keys are added.
    Implementing: Merge Logic -> __init__
    """
    # Logic: Merge self.metadata with new metadata
    # Logic: Merge self.tables with new tables
    # Return RDB(metadata=merged_meta, tables=merged_tables)
```

### 3.2 Convenience Wrappers

| Method | Purpose | Implementation Strategy |
| :--- | :--- | :--- |
| **`update_table`** | Update/Replace a **single** table. | Wraps `update_tables({name: df}, {name: schema})`. |
| **`add_table`** | Add a **single** NEW table. | Checks `if name in self.table_names`: Error. Then calls `update_tables`. |

### 3.3 New Utilities & Logic Updates

1.  **Relax `validate_key_consistency`**:
    -   **Goal**: Allow validation to proceed even if the referenced parent table is missing from the current RDB subset.
    -   **Change**: Check existence of parent table in `self.table_names` before attempting to access it. Log warning for missing parents.

2.  **`RDB.canonicalize_key_types`**:
    -   **Goal**: Ensure all Primary and Foreign Keys are converted to string type for consistency.
    -   **Constraint**: Throw error for "dangerous" conversions (e.g. floats).
    -   **Implementation**:
        -   Rename `fastdfs/utils/type_inference.py` to `fastdfs/utils/type_utils.py`.
        -   Add `safe_convert_to_string` utility to `type_utils.py`.
        -   Add `canonicalize_key_types(self) -> RDB` method.
        -   Iterate all ID columns (PKs, FKs), convert using utility, call `update_tables`.

## 4. Refactor Implementation Plan

### 4.1 Datatset Class (`fastdfs/dataset/rdb.py`)
1.  **Remove** methods:
    -   `create_new_with_tables`
    -   `create_new_with_tables_and_metadata`
2.  **Add** new methods:
    -   `update_tables(tables=None, metadata=None)` (Core)
    -   `update_table(name, dataframe, ...)` (Wrapper)
    -   `add_table(name, dataframe, ...)` (Wrapper, updated logic)
    -   `canonicalize_key_types()` (New Transformation)
3.  **Update** `validate_key_consistency`:
    -   Add check for parent table existence.

### 4.2 Migration of References
All calls to the old API will be replaced with `update_tables`.

| File | Context | Old Call | New Call |
| :--- | :--- | :--- | :--- |
| `fastdfs/dataset/rdb.py` | `RDB.add_table` | `self.create_new_with_tables_and_metadata(...)` | `self.update_tables(tables=..., metadata=...)` |
| `fastdfs/transform/base.py` | `ColumnTransform.apply` | `rdb.create_new_with_tables_and_metadata(...)` | `rdb.update_tables(tables=new_tables, metadata=updated_metadata)` |
| `fastdfs/transform/dummy_table.py` | `DummyTableTransform.apply` | `dataset.create_new_with_tables_and_metadata(...)` | `dataset.update_tables(tables=new_tables, metadata=new_metadata)` |
| `fastdfs/transform/fill_missing_pk.py` | `FillMissingPK.apply` | `rdb.create_new_with_tables_and_metadata(...)` | `rdb.update_tables(tables=new_tables, metadata=new_metadata)` |
| `fastdfs/transform/infer_schema.py` | `InferSchemaTransform.apply` | `rdb.create_new_with_tables_and_metadata(...)` | `rdb.update_tables(tables=new_tables, metadata=new_table_schemas)` |

### 4.3 Test Updates
Tests verifying the calls (checking mocks) need to be updated.

| Test File | Change |
| :--- | :--- |
| `tests/test_rdb.py` | Replace `rdb.create_new_with_tables(...)` tests with `rdb.update_tables(...)`. |
| `tests/test_dummy_table_transform.py` | Update `mock_dataset.create_new_with_tables_and_metadata.assert_called...` to `mock_dataset.update_tables.assert_called...`. |
| `tests/test_fill_missing_primary_key.py` | Update mock assertion from `create_new...` to `update_tables`. |

## 5. Reference: Analysis of Single vs Bulk Updates

*(From previous analysis)*

### The Cost of Single-Item Updates (Looping)
If we were to remove bulk update capability and force transforms to loop over `update_table`:
-   **Data Cost**: Negligible (Pandas DataFrames are passed by reference).
-   **Overhead**: `O(N)` Pydantic schema validations and `O(N)` dictionary copies for N tables.
-   **Ergonomics**: Transforms producing batch updates would need to implement loops, handling potential partial failures.

### Why Bulk `update_tables` is the Core
-   **Efficiency**: One validation step, one object creation for N changes.
-   **Atomicity**: The RDB state moves from `State A` -> `State B` in one step, even if multiple tables change.
-   **Flexibility**: It handles both implementation details (like `InferSchemaTransform` rewriting all schemas) and simple user updates.

### Conclusion
We retain the "Bulk Update" capability via `update_tables` as the primary engine method, and offer `update_table` for user convenience.

## 6. Implementation Checklist (TODO)

- [ ] **Step 1: Utilities Refactor**
    - [ ] Rename `fastdfs/utils/type_inference.py` -> `fastdfs/utils/type_utils.py`.
    - [ ] Add `safe_convert_to_string` to `type_utils.py`.
    - [ ] Update imports in `fastdfs/dataset/rdb.py`, `fastdfs/transform/*.py`, tests.

- [ ] **Step 2: RDB Core Refactor**
    - [ ] Implement `RDB.update_tables` using `__init__`.
    - [ ] Implement `RDB.update_table`.
    - [ ] Implement `RDB.add_table` (update existing).
    - [ ] Remove `create_new_with_tables...` methods.
    - [ ] Update `validate_key_consistency` (relax missing parent check).
    - [ ] Implement `RDB.canonicalize_key_types`.

- [ ] **Step 3: Update Call Sites**
    - [ ] Refactor `fastdfs/transform/base.py`.
    - [ ] Refactor `fastdfs/transform/dummy_table_transform.py`.
    - [ ] Refactor `fastdfs/transform/fill_missing_pk.py`.
    - [ ] Refactor `fastdfs/transform/infer_schema.py`.

- [ ] **Step 4: Verify Tests**
    - [ ] Fix/Update `tests/test_rdb.py`.
    - [ ] Fix/Update `tests/test_dummy_table_transform.py`.
    - [ ] Fix/Update `tests/test_fill_missing_primary_key.py`.
    - [ ] Add unit tests for `RDB.update_table` and `RDB.update_tables`.
    - [ ] Add unit tests for `RDB.canonicalize_key_types` (valid and error cases).
    - [ ] Add unit tests for `safe_convert_to_string`.
    - [ ] Run full test suite.

- [ ] **Step 5: Documentation**
    - [ ] Update documentation for `RDB` class changes (APIs).
    - [ ] Update User Guide if necessary (showing `canonicalize_key_types` or `update_tables` usage).
