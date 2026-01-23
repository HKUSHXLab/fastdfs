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

## 6. Execution & Deviations (Jan 23, 2026)

During implementation, several refinements were made to enforce stricter data consistency and type safety.

### 6.1 Strict `update_tables` API
Instead of optional arguments (`tables=None, metadata=None`), the implementation now enforces:
-   **Mandatory Arguments**: Both `tables` and `metadata` must be provided (dictionaries, can be empty).
-   **Key Consistency**: `set(tables.keys())` must exactly match `set(metadata.keys())`. This prevents "widowed" data (tables without schemas) or "ghost" metadata.
-   **Schema Validation**: The columns in the provided DataFrame are checked against the provided RDBTableSchema. Any column in the DataFrame missing from the Schema raises a `ValueError`.

### 6.2 Strict `update_table` API
-   **Mandatory Schema**: The `schema` argument is now required.
-   **Reasoning**: Implicit behavior (inferring schemas or fetching existing ones inside a method that implies mutation) was deemed too fragile used in an immutable-style API. The caller must explicit provide the schema describing the new state of the table.

### 6.3 Validation Logic Polish
-   **FK Group Validation**: `validate_key_consistency` groups Foreign Keys by their target (parent). Even if the parent table is missing (common in partial loads), the validator checks consistency *among* the sibling children.
-   **Strict Float Rejection**: The `safe_convert_to_string` utility used by `canonicalize_key_types` explicitly raises a `ValueError` for floating-point types to prevent dangerous ID conversions (e.g. `10.0` -> `"10.0"` instead of `"10"`).

### 6.4 Test Restructuring
-   Unit tests for utilities were moved to `tests/utils/`.
-   New API tests were merged into `tests/test_rdb.py`, which was reorganized into class-based suites (`TestRDBConstruction`, `TestRDBMutation`, etc.).


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

## 6. Implementation Checklist (Done)

- [x] **Step 1: Utilities Refactor**
    - [x] Rename `fastdfs/utils/type_inference.py` -> `fastdfs/utils/type_utils.py`.
    - [x] Add `safe_convert_to_string` to `type_utils.py`.
    - [x] Update imports in `fastdfs/dataset/rdb.py`, `fastdfs/transform/*.py`, tests.

- [x] **Step 2: RDB Core Refactor**
    - [x] Implement `RDB.update_tables` using `__init__`.
    - [x] Implement `RDB.update_table`.
    - [x] Implement `RDB.add_table` (update existing).
    - [x] Remove `create_new_with_tables...` methods.
    - [x] Update `validate_key_consistency` (relax missing parent check).
    - [x] Implement `RDB.canonicalize_key_types`.

- [x] **Step 3: Update Call Sites**
    - [x] Refactor `fastdfs/transform/base.py`.
    - [x] Refactor `fastdfs/transform/dummy_table_transform.py`.
    - [x] Refactor `fastdfs/transform/fill_missing_pk.py`.
    - [x] Refactor `fastdfs/transform/infer_schema.py`.

- [x] **Step 4: Verify Tests**
    - [x] Fix/Update `tests/test_rdb.py`.
    - [x] Fix/Update `tests/test_dummy_table_transform.py`.
    - [x] Fix/Update `tests/test_fill_missing_primary_key.py`.
    - [x] Add unit tests for `RDB.update_table` and `RDB.update_tables`.
    - [x] Add unit tests for `RDB.canonicalize_key_types` (valid and error cases).
    - [x] Add unit tests for `safe_convert_to_string`.
    - [x] Run full test suite.

- [x] **Step 5: Documentation**
    - [x] Update documentation for `RDB` class changes (APIs).
    - [x] Update User Guide if necessary (showing `canonicalize_key_types` or `update_tables` usage).
