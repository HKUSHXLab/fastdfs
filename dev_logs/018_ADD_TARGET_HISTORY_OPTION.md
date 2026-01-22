# Dev Plan: Dynamic RDB Augmentation (Add Table)

## 1. Goal
Implement a generic mechanism to add new tables to an existing `RDB` object. This empowers users to dynamically augment the relational structure (e.g., adding the target dataframe as a historical table) manually before passing it to the DFS pipeline.

## 2. Motivation
Users often need to augment the RDB with auxiliary data (like target history) to generate richer features. For example, in a recommendation system, the target dataframe might contain `(user_id, item_id, timestamp, rating)`. To predict a future rating, it's highly valuable to have features like "average of user's past ratings" or "number of items rated by user in the last week". Without adding the target history back into the RDB, the DFS engine cannot see these past interactions.

By providing a generic `add_table` method, we give users full control to manually add this `__target_history__` table (or any other auxiliary data) to the RDB structure. This approach is cleaner than building "magic" helpers and decouples data preparation from the DFS engine.

## 3. High-Level Design

### 3.1 New RDB Method
Add a method to the `RDB` class to genericly add a table.

```python
class RDB:
    def add_table(
        self, 
        dataframe: pd.DataFrame, 
        name: str, 
        time_column: Optional[str] = None,
        primary_key: Optional[str] = None,
        foreign_keys: Optional[List[Tuple[str, str, str]]] = None,
        column_types: Optional[Dict[str, str]] = None
    ) -> 'RDB'
```

*   **Immutability**: Returns a *new* `RDB` instance (sharing unchanged data) to avoid side effects on the original dataset.
*   **Schema Inference**: Automatically infers column types (Float, DateTime, etc.) from the dataframe.
*   **Metadata Update**: Updates the internal `RDBMeta` with the new table schema and relationships.
*   **Column Types**: Allows optional manual override of column types (e.g., to force a column to be treated as categorical).

## 4. Implementation Details

### 4.1 RDB Class (`fastdfs/dataset/rdb.py`)
*   Implement `add_table`.
    *   **Inputs**: 
        *   `dataframe`: The data to add.
        *   `name`: Name of the new table.
        *   `time_column`: Name of the time index column.
        *   `primary_key`: Name of the primary key column (optional).
        *   `foreign_keys`: List of `(child_col, parent_table, parent_col)`.
        *   `column_types`: Dict mapping column names to `RDBColumnDType` strings (optional).
    *   **Process**:
        1.  **Validation**: Check if table name exists. Validate FK targets exist.
        2.  **Schema Creation**: 
            *   Inspect dataframe dtypes.
            *   Override with `column_types` if provided.
            *   Set `primary_key` dtype if applicable.
            *   Set `foreign_key` dtype for columns in `foreign_keys`.
        3.  **New Instance**:
            *   Create new `RDBTableSchema`.
            *   Merge with existing metadata.
            *   Return new `RDB` instance.

## 5. Usage Example

```python
# User manually prepares the RDB with target history
rdb_augmented = rdb.add_table(
    dataframe=target_df, 
    name="__target_history__",
    time_column="time",
    foreign_keys=[
        ("user_id", "users", "user_id")
    ]
)

# Run DFS using the augmented RDB
features = compute_dfs_features(
    rdb=rdb_augmented,
    target_dataframe=target_df,
    key_mappings={"user_id": "users.user_id"},
    cutoff_time_column="time"
)

# Result: Features like 'users.MEAN(__target_history__.target_val)' are generated.
```

## 6. Safety & Validation
*   **Duplicate Names**: Raise `ValueError` if `name` already exists in `rdb.table_names`.
*   **Invalid Foreign Keys**: Raise `ValueError` if `parent_table` or `parent_col` does not exist.
*   **Type Compatibility**: Ensure inferred types are valid `RDBColumnDType` values.

## 7. Action Plan

- [ ] Implement `RDB.add_table` method in `fastdfs/dataset/rdb.py`.
- [ ] Add unit tests for `add_table` in `tests/test_rdb.py`:
    - Test success case with inferred types.
    - Test success case with manual `column_types` override.
    - Test error case: Duplicate table name.
- [ ] Create an integration test `tests/test_target_history.py` that simulates the "target history" flow:
    - Create RDB.
    - Create Target DF.
    - Augment RDB with Target DF using `add_table`.
    - Run DFS and verify features like `MEAN(__target_history__.value)` are generated.
- [ ] Update `docs/api_reference.md` to document the new `add_table` method.
- [ ] Update `docs/user_guide.md` with a "Target History" section or "Advanced RDB Manipulation" section showing the example.
