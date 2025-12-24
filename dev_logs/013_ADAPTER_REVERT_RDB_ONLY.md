# Adapter Subpackage Revert (RDB Only)

## Status
Completed.

## Changes
- Reverted `RelBenchAdapter.load` to only load the RDB content.
- Removed task loading logic (target DataFrame, key mappings, cutoff time).
- `load` now returns only `RDBDataset`.
- Updated `examples/test_relbench_load.py` to verify the simplified API.

## Verification
- Ran `examples/test_relbench_load.py` and confirmed that `RDBDataset` is loaded correctly without task info.
