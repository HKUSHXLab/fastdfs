"""
Basic usage example demonstrating the FastDFS table-centric DFS API.

This example shows how to:
1. Load an RDB dataset
2. Create a transform pipeline  
3. Apply DFS feature generation to a target dataframe
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastdfs
from fastdfs.transform import (
    RDBTransformPipeline,
    HandleDummyTable,
    FeaturizeDatetime,
    FilterColumn,
    RDBTransformWrapper,
    FillMissingPrimaryKey,
    CanonicalizeTypes,
)
from fastdfs.utils.logging_config import configure_logging

# Configure logging
configure_logging(level="INFO")

def main():
    print("=== FastDFS Basic Usage Example ===")
    
    # 1. Load RDB dataset
    rdb_path = project_root / "tests" / "data" / "test_rdb_new"
    print(f"Loading RDB from: {rdb_path}")
    rdb = fastdfs.load_rdb(str(rdb_path))
    print(f"Loaded dataset: {rdb.metadata.name}")
    print(f"Tables: {rdb.table_names}")
    
    # 2. Create transform pipeline following the design document
    transform_pipeline = RDBTransformPipeline([
        HandleDummyTable(),
        FillMissingPrimaryKey(),
        RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
        RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
        RDBTransformWrapper(CanonicalizeTypes())
    ])
    
    # 3. Create target dataframe (following test_dfs_engines.py pattern)
    target_df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2],
        "item_id": [1, 2, 3, 2, 1], 
        "timestamp": pd.to_datetime([
            "2023-01-01 10:00:00",
            "2023-01-02 11:00:00", 
            "2023-01-03 12:00:00",
            "2023-01-04 13:00:00",
            "2023-01-05 14:00:00"
        ])
    })
    print(f"Target dataframe shape: {target_df.shape}")
    
    # 4. Apply transforms to RDB
    print("Applying transforms...")
    transformed_rdb = transform_pipeline(rdb)
    
    # 5. Generate features using Featuretools engine
    print("Generating features with Featuretools...")
    features_df = fastdfs.compute_dfs_features(
        rdb=transformed_rdb,
        target_dataframe=target_df,
        key_mappings={
            "user_id": "user.user_id",
            "item_id": "item.item_id"
        },
        cutoff_time_column="timestamp",
        config_overrides={
            "max_depth": 2,
            "engine": "featuretools",
            "agg_primitives": ["count", "mean", "max", "min"]
        }
    )
    
    print(f"Generated {len(features_df.columns)} total columns")
    print(f"Original columns: {len(target_df.columns)}")
    print(f"New feature columns: {len(features_df.columns) - len(target_df.columns)}")
    print(f"Result shape: {features_df.shape}")
    print("\nExample features:")
    for col in features_df.columns:
        if col not in target_df.columns:
            print(f"  - {col}")
    
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
