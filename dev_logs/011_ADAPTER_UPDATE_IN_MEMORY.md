# Adapter Subpackage Update

## Status
Completed.

## Changes
- Modified `fastdfs/dataset/rdb.py` to support in-memory initialization of `RDBDataset` (via `metadata` and `tables` arguments).
- Updated `fastdfs/adapter/relbench.py`:
    - Replaced `logging` with `loguru`.
    - Added `load(task_name, split)` method to load dataset and task in-memory.
    - `load` returns `(rdb, target_df, key_mappings, cutoff_time_column)`.
    - `convert` method is preserved but now uses `loguru`.

## Verification
- Created `examples/test_relbench_load.py` to verify `load` method with `rel-f1` dataset and `driver-position` task.
- Verified that `RDBDataset` is created correctly and `target_df`, `key_mappings`, `cutoff_time_column` are correct.
