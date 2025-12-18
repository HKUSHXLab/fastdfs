# PR: Multi-key Support, New Transforms, and Engine Robustness

## Summary
This PR introduces support for multiple key mappings in DFS, adds new RDB transformations (`CanonicalizeTypes`, updated `FeaturizeDatetime`), and improves the robustness of the DFS engines against complex target dataframes and CI environment issues.

## Changes

### 1. Engine Robustness & Optimization
- **Target DataFrame Filtering**: Modified `BaseEngine` to filter the `target_dataframe` before passing it to the underlying engines (Featuretools/DFS2SQL). It now only retains the key columns, time index, and index. This prevents crashes when the target dataframe contains columns with complex data types (e.g., lists, dictionaries) that are not involved in feature generation but cause issues during internal processing or type inference.
- **DFS2SQL Empty Result Handling**: Fixed `DFS2SQLEngine` to return an empty DataFrame with the correct index instead of `None` when no features are generated, ensuring consistency with the API contract.
- **DuckDB Concurrency**: Implemented random temporary file path generation for DuckDB connections in `DFS2SQLEngine` to prevent file locking collisions when running tests in parallel or CI environments.
- **SQLGlot Compatibility**: Fixed a `KeyError` in `gen_sqls.py` by safely accessing the "from" key in SQLGlot expressions.

### 2. Multi-Key Support
- **Iterative Key Processing**: Restored and fixed the logic in `base_engine.py` to handle multiple key mappings. The engine now iterates through each key mapping to generate features sequentially. This resolves issues where Featuretools could not handle multiple entity paths simultaneously or correctly.

### 3. New Transformations
- **CanonicalizeTypes**: Added a new `CanonicalizeTypes` transform. This transform enforces data types on the dataframe based on the provided metadata. It drops columns not defined in the metadata and raises an error if metadata-defined columns are missing.
- **FeaturizeDatetime Update**: Updated the `FeaturizeDatetime` transform to include a `retain_original` parameter (default: `True`). This allows users to choose whether to keep the original datetime column after generating sub-features (year, month, etc.).

### 4. Testing & CI
- **Complex Target Columns**: Added `tests/test_complex_target_columns.py` to verify that engines can gracefully handle target dataframes with complex column types.
- **Type Canonicalization**: Added `tests/test_type_transform.py` to test the `CanonicalizeTypes` transform.
- **Example Testing**: Added `tests/test_examples.py` to include `examples/basic_usage.py` in the CI pipeline, ensuring the documentation example remains valid.
- **Transform Updates**: Updated `tests/test_transforms.py` to verify the `retain_original` functionality in `FeaturizeDatetime`.

## Impact
- **Reliability**: The DFS process is now more robust to "dirty" target dataframes containing non-primitive types.
- **Flexibility**: Users have more control over datetime feature generation and type enforcement.
- **Correctness**: Multi-key relationships are now handled correctly, and edge cases (no features generated) return consistent types.
