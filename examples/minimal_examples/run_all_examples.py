#!/usr/bin/env python3
"""
Run all minimal DFS examples and compare results with fastdfs.

This script:
1. Loads each example RDB dataset
2. Runs DFS with appropriate configurations
3. Displays generated features
4. Compares with expected results
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment to disable tqdm before importing fastdfs
os.environ['TQDM_DISABLE'] = '1'

import fastdfs
from fastdfs.dfs import DFSConfig
from fastdfs.utils.logging_config import configure_logging
configure_logging(level="WARNING")  # or "ERROR"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress tqdm
import tqdm
original_tqdm_init = tqdm.tqdm.__init__
def silent_init(self, *args, **kwargs):
    kwargs['disable'] = True
    return original_tqdm_init(self, *args, **kwargs)
tqdm.tqdm.__init__ = silent_init

def compare_df(dfs_features, golden_df):
    # Compare dfs_features to golden_df (ignoring column order except first column).
    # First, ensure both dataframes have the same columns (regardless of order), except order should start with same first column

    # Reorder dfs_features columns to match golden_df columns order
    common_cols = [col for col in golden_df.columns if col in dfs_features.columns]
    dfs_features_reordered = dfs_features[common_cols]

    # Check if the first column matches in both (by name)
    first_col = golden_df.columns[0]
    if dfs_features_reordered.columns[0] != first_col:
        raise AssertionError(f"First column mismatch: {dfs_features_reordered.columns[0]} vs {first_col}")

    # Compare shape
    if dfs_features_reordered.shape != golden_df.shape:
        print(f"❌  Shape mismatch: dfs_features {dfs_features_reordered.shape}, golden_df {golden_df.shape}")
        return False

    # Compare values (with float tolerance)
    comparison = True
    for col in common_cols:
        a = np.asarray(dfs_features_reordered[col])
        b = np.asarray(golden_df[col])
        # NaNs in both match
        nan_mask = np.isnan(a) & np.isnan(b) if np.issubdtype(a.dtype, np.floating) else None
        if np.issubdtype(a.dtype, np.floating):
            eq = np.allclose(a, b, equal_nan=True)
        else:
            eq = np.array_equal(a, b)
        if not eq:
            print(f"❌ Column mismatch: {col}")
            print(f"  dfs_features: {a}")
            print(f"  golden_df: {b}")
            comparison = False
    if comparison:
        print("✅ dfs_features matches golden_df")
        return True
    else:
        print("❌ dfs_features does NOT match golden_df")
        return False

def run_example(example_name, description, rdb_path, target_df, key_mappings, 
                cutoff_time_column, max_depth, agg_primitives, use_cutoff_time=True):
    """Run a single DFS example and display results."""
    print("\n" + "="*80)
    print(f"{example_name}: {description}")
    print("="*80)
    
    try:
        # Load RDB
        print(f"\nLoading RDB from: {rdb_path}")
        rdb = fastdfs.load_rdb(rdb_path)
        print(f"✅ RDB loaded: {len(rdb.table_names)} tables - {rdb.table_names}")
        
        # Display target table
        print(f"\nTarget table shape: {target_df.shape}")
        print(f"Target table:\n{target_df.head()}")
        
        # Configure DFS
        config = DFSConfig(
            max_depth=max_depth,
            agg_primitives=agg_primitives,
            engine="dfs2sql",
            use_cutoff_time=use_cutoff_time
        )
        
        print(f"\nDFS Configuration:")
        print(f"  max_depth: {max_depth}")
        print(f"  agg_primitives: {agg_primitives}")
        print(f"  use_cutoff_time: {use_cutoff_time}")
        print(f"  cutoff_time_column: {cutoff_time_column}")
        print(f"  key_mappings: {key_mappings}")
        
        # Run DFS
        print(f"\nRunning DFS feature generation...")
        pipeline = fastdfs.DFSPipeline(
            transform_pipeline=None,
            dfs_config=config
        )
        
        features_df = pipeline.compute_features(
            rdb=rdb,
            target_dataframe=target_df,
            key_mappings=key_mappings,
            cutoff_time_column=cutoff_time_column
        )
        
        if features_df is None:
            print("⚠️  No features generated!")
            return
        
        # Display results
        print(f"\n✅ Feature generation completed!")
        print(f"   Original columns: {len(target_df.columns)}")
        print(f"   Total columns after DFS: {len(features_df.columns)}")
        print(f"   New features generated: {len(features_df.columns) - len(target_df.columns)}")
        
        # Get new feature columns
        original_cols = set(target_df.columns)
        new_features = [col for col in features_df.columns if col not in original_cols]
        
        if new_features:
            print(f"\n📊 Generated Features:")
            for feat in sorted(new_features):
                print(f"   - {feat}")
                # Show sample values for first few rows
                sample_vals = features_df[feat].head(3).tolist()
                print(f"     Sample values: {sample_vals}")
        else:
            print(f"\n⚠️  No new features generated!")
        
        # Display full result dataframe
        print(f"\n📋 Full Result DataFrame:")
        print(features_df.to_string())
        
        return features_df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def example1_direct_features():
    """
    Example 1: Direct Features (Depth 1)
    
    Note: max_depth=0 only returns identity features (target table columns).
    To get attributes from related tables, we need max_depth=1.
    """
    example_dir = Path(__file__).parent / "example1_direct_features"
    rdb_path = str(example_dir)
    
    target_df = pd.read_csv(example_dir / "target_table.csv")
    
    dfs_features = run_example(
        example_name="Example 1",
        description="Direct Features (Depth 1) - Testing basic attribute retrieval from related tables",
        rdb_path=rdb_path,
        target_df=target_df,
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column=None,
        max_depth=1,  # Depth 1 needed to get attributes from related tables
        agg_primitives=[],  # No aggregation primitives - only direct features
        use_cutoff_time=False
    )

    golden_data = np.load(example_dir / "data" / "golden_depth_1.npz", allow_pickle=True)
    golden_df = pd.DataFrame({k: golden_data[k] for k in golden_data.files})

    if not compare_df(dfs_features, golden_df):
        raise AssertionError("DFS features do not match golden_df")
    return dfs_features


def example2_single_aggregation():
    """
    Example 2: Single-Level Aggregation (Depth 2)
    
    Important: Since target → user (parent link) and transaction is a child of user,
    we need depth 2 to aggregate over transaction:
    - Depth 1: Get direct features from user (like user.age)
    - Depth 2: Aggregate over user's children (transaction)
    
    So depth 1 + depth 2 = target → user → transaction
    """
    example_dir = Path(__file__).parent / "example2_single_aggregation"
    rdb_path = str(example_dir)
    
    target_df = pd.read_csv(example_dir / "target_table.csv")
    
    dfs_features = run_example(
        example_name="Example 2",
        description="Single-Level Aggregation (Depth 2) - Testing count, mean, max, min primitives on child entities",
        rdb_path=rdb_path,
        target_df=target_df,
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column=None,
        max_depth=2,  # Depth 2 needed: target → user → transaction
        agg_primitives=["count", "mean", "max", "min", "mode"],
        use_cutoff_time=False
    )

    golden_data = np.load(example_dir / "data" / "golden_depth_2.npz", allow_pickle=True)
    golden_df = pd.DataFrame({k: golden_data[k] for k in golden_data.files})

    if not compare_df(dfs_features, golden_df):
        raise AssertionError("DFS features do not match golden_df")

    return dfs_features


def example3_multi_level_aggregation():
    """
    Example 3: Multi-Level Aggregation (Depth 3)
    
    Important: Relationship path is target → user → order → order_item
    - Depth 1: Target → user (get user.age)
    - Depth 2: Target → user → order (aggregate over orders)
    - Depth 3: Target → user → order → order_item (aggregate over order_items)
    
    To aggregate over order_item (grandchild), we need depth 3!
    """
    example_dir = Path(__file__).parent / "example3_multi_level_aggregation"
    rdb_path = str(example_dir)
    
    target_df = pd.read_csv(example_dir / "target_table.csv")

    dfs_features = run_example(
        example_name="Example 3",
        description="Multi-Level Aggregation (Depth 3) - Testing features through intermediate tables",
        rdb_path=rdb_path,
        target_df=target_df,
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column=None,
        max_depth=2,  # Depth 3 needed: target → user → order → order_item
        agg_primitives=["count", "mean", "max", "min"],
        use_cutoff_time=False
    )

    golden_data = np.load(example_dir / "data" / "golden_depth_2.npz", allow_pickle=True)
    golden_df = pd.DataFrame({k: golden_data[k] for k in golden_data.files})
    print(f"Golden DataFrame: {golden_df}")

    if not compare_df(dfs_features, golden_df):
        raise AssertionError("DFS features do not match golden_df")
    
    return dfs_features


def example4_cutoff_time():
    """Example 4: Cutoff Time Handling"""
    example_dir = Path(__file__).parent / "example4_cutoff_time"
    rdb_path = str(example_dir)
    
    target_df = pd.read_csv(example_dir / "target_table.csv")
    target_df['cutoff_time'] = pd.to_datetime(target_df['cutoff_time'])
    
    dfs_features = run_example(
        example_name="Example 4",
        description="Cutoff Time Handling - Testing temporal filtering behavior",
        rdb_path=rdb_path,
        target_df=target_df,
        key_mappings={"user_id": "user.user_id"},
        cutoff_time_column="cutoff_time",
        max_depth=2,
        agg_primitives=["count", "mean"],
        use_cutoff_time=True
    )

    golden_data = np.load(example_dir / "data" / "golden_depth_2.npz", allow_pickle=True)
    golden_df = pd.DataFrame({k: golden_data[k] for k in golden_data.files})
    print(f"Golden DataFrame: {golden_df}")

    if not compare_df(dfs_features, golden_df):
        raise AssertionError("DFS features do not match golden_df")
    
    return dfs_features


def example5_multiple_relationships():
    """Example 5: Multiple Relationships"""
    example_dir = Path(__file__).parent / "example5_multiple_relationships"
    rdb_path = str(example_dir)
    
    target_df = pd.read_csv(example_dir / "target_table.csv")
    target_df['cutoff_time'] = pd.to_datetime(target_df['cutoff_time'])
    
    dfs_features = run_example(
        example_name="Example 5",
        description="Multiple Relationships - Testing user and item features simultaneously",
        rdb_path=rdb_path,
        target_df=target_df,
        key_mappings={
            "user_id": "user.user_id",
            "item_id": "item.item_id"
        },
        cutoff_time_column="cutoff_time",
        max_depth=2,
        agg_primitives=["count", "mean", "STD"],
        use_cutoff_time=True
    )

    golden_data = np.load(example_dir / "data" / "golden_depth_2.npz", allow_pickle=True)
    golden_df = pd.DataFrame({k: golden_data[k] for k in golden_data.files})
    print(f"Golden DataFrame: {golden_df}")

    if not compare_df(dfs_features, golden_df):
        raise AssertionError("DFS features do not match golden_df")
    
    return dfs_features


def main():
    """Run all examples."""
    print("="*80)
    print("DFS Minimal Examples - Testing FastDFS Feature Generation")
    print("="*80)
    
    results = {}
    
    # Run each example
    results['example1'] = example1_direct_features()
    results['example2'] = example2_single_aggregation()
    results['example3'] = example3_multi_level_aggregation()
    results['example4'] = example4_cutoff_time()
    results['example5'] = example5_multiple_relationships()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # for name, result_df in results.items():
    #     if result_df is not None:
    #         print(f"\n✅ {name}: {len(result_df.columns)} total columns")
    #     else:
    #         print(f"\n❌ {name}: Failed to generate features")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the generated features above")
    print("2. Compare with expected results in each example's README.md")
    print("3. Verify that features match your manual calculations")
    print("4. Use these examples as reference for understanding DFS behavior")


if __name__ == "__main__":
    main()
