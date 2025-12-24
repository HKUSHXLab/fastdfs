# Adapter Subpackage Update (Load All Splits)

## Status
Completed.

## Changes
- Modified `RelBenchAdapter.load` to return `(rdb, train_df, val_df, test_df, key_mappings, cutoff_time_column)`.
- Updated `examples/test_relbench_load.py` to verify the new API.

## Verification
- Ran `examples/test_relbench_load.py` and confirmed that train, val, and test DataFrames are loaded correctly.
