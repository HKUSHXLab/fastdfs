# Refactor: RDBDataset -> RDB

## Status
Completed.

## Changes
- Renamed `RDBDataset` to `RDB` and `RDBDatasetMeta` to `RDBMeta` in docstrings and documentation.
- Updated `fastdfs/dataset/rdb.py`, `fastdfs/dataset/meta.py`, `fastdfs/api.py`, `fastdfs/adapter/relbench.py`.
- Updated `docs/user_guide.md` and `README.md`.
- Removed "simplified dataset without tasks" phrasing to reduce cognitive load.

## Rationale
- `fastdfs` focuses on the relational database structure (RDB) rather than a machine learning "dataset" concept (which implies splits, targets, labels, etc.).
- The new naming is cleaner and more accurate.
