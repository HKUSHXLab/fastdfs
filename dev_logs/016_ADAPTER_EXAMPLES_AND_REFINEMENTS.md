# Adapter Examples and Refinements

## Status
Completed.

## Summary
This phase focused on creating end-to-end examples for RelBench and DBInfer, refining the `RDB` API for better usability, and integrating the `dbinfer_bench` library directly into the codebase.

## Changes

### 1. DBInfer Integration
- **Adapter**: Implemented `DBInferAdapter` in `fastdfs/adapter/dbinfer.py`.
- **Source Integration**: Integrated `dbinfer_bench` source code into `fastdfs/dbinfer_bench/` to remove external dependency issues and allow for customization.
- **Licensing**: Added Apache 2.0 license and attribution headers to the integrated `dbinfer_bench` files.

### 2. Example Scripts
- **RelBench**: Created `examples/relbench_f1_example.py`.
  - Demonstrates loading `rel-f1`, retrieving tasks via `relbench.tasks.get_task`, and running the DFS pipeline.
  - Fixed issues with task dataframe access and internal API calls.
- **DBInfer**: Created `examples/dbinfer_diginetica_example.py`.
  - Demonstrates loading `diginetica` dataset.
  - Implemented multi-key mapping (`queryId` -> `Query`, `itemId` -> `Product`) for richer feature generation.
  - Added robust handling for missing tables and timestamp type conversion.

### 3. API & Core Refinements
- **RDB Usability**:
  - Added `RDB.get_table(name)` as an alias for `get_table_dataframe(name)`.
  - Added `RDBTableSchema.primary_key` property for quick access to the PK column name.
- **Type Handling**:
  - Updated `CanonicalizeTypes` transform to map `category_t` columns to `string` (object) dtype instead of pandas Categoricals. This improves compatibility with downstream tools that expect string identifiers.
- **Logging**:
  - Enhanced `RDBTransformPipeline` to log the name of each transform being executed (including unwrapping `RDBTransformWrapper` names).

## Verification
- **RelBench**: `examples/relbench_f1_example.py` runs successfully, generating features for the `driver-dnf` task.
- **DBInfer**: `examples/dbinfer_diginetica_example.py` runs successfully (verified with mock environment), generating features for both Query and Product entities.
- **Tests**: Updated `tests/test_type_transform.py` passed.
