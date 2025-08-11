#!/usr/bin/env python3
"""
FastDFS Python API Example

This example demonstrates how to use the FastDFS Python API for automated
feature engineering using Deep Feature Synthesis (DFS).

Usage:
    python python_api_example.py [--dataset-path PATH] [--output-path PATH] [--max-depth DEPTH] [--engine ENGINE]

Examples:
    # Use default test dataset
    python python_api_example.py
    
    # Use custom dataset path
    python python_api_example.py --dataset-path /path/to/your/dataset
    
    # Configure DFS parameters
    python python_api_example.py --max-depth 3 --engine dfs2sql
"""

import argparse
from loguru import logger
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import fastdfs
from fastdfs.utils.logging_config import configure_logging

# Configure logging for the example
configure_logging(level="INFO")


def main():
    """Main function to demonstrate FastDFS API usage."""
    parser = argparse.ArgumentParser(
        description="FastDFS Python API Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --dataset-path /path/to/dataset
  %(prog)s --max-depth 3 --engine dfs2sql
        """
    )
    
    # Default to test dataset relative to project root
    default_dataset = project_root / "tests" / "data" / "test_rdb"
    
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=default_dataset,
        help=f"Path to dataset directory (default: {default_dataset})"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output directory path (default: temporary directory)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for DFS (default: 2)"
    )
    
    parser.add_argument(
        "--engine",
        choices=["featuretools", "dfs2sql"],
        default="dfs2sql",
        help="DFS engine to use (default: dfs2sql)"
    )
    
    parser.add_argument(
        "--operation",
        choices=["load", "transform", "dfs", "full-pipeline"],
        default="full-pipeline",
        help="Operation to perform (default: full-pipeline)"
    )
    
    parser.add_argument(
        "--transform-type",
        choices=["pre-dfs", "post-dfs", "single"],
        default="pre-dfs",
        help="Transform type for transform operation (default: pre-dfs)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not args.dataset_path.exists():
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        return False
    
    # Set up output path
    if args.output_path is None:
        temp_dir = tempfile.mkdtemp(prefix="fastdfs_example_")
        output_path = Path(temp_dir) / "output"
        cleanup_temp = True
        logger.info(f"Using temporary output directory: {output_path}")
    else:
        output_path = args.output_path
        cleanup_temp = False
    
    try:
        logger.info("=== FastDFS Python API Example ===")
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Operation: {args.operation}")
        
        if args.operation == "load":
            demonstrate_load_dataset(args.dataset_path)
        elif args.operation == "transform":
            demonstrate_transform(args.dataset_path, output_path, args.transform_type)
        elif args.operation == "dfs":
            demonstrate_dfs(args.dataset_path, output_path, args.max_depth, args.engine)
        elif args.operation == "full-pipeline":
            demonstrate_full_pipeline(args.dataset_path, output_path, args.max_depth, args.engine)
        
        logger.info("=== Example completed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory if used
        if cleanup_temp and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def demonstrate_load_dataset(dataset_path: Path):
    """Demonstrate loading a dataset and inspecting its structure."""
    logger.info("--- Loading Dataset ---")
    
    dataset = fastdfs.load_dataset(dataset_path)
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Dataset name: {dataset.metadata.dataset_name}")
    logger.info(f"Number of tables: {len(dataset.metadata.tables)}")
    
    for table in dataset.metadata.tables:
        logger.info(f"  Table: {table.name}")
        logger.info(f"    Columns: {len(table.columns)}")
        logger.info(f"    Format: {table.format}")
        logger.info(f"    Source: {table.source}")
        
        # Show column details
        for col in table.columns:
            logger.info(f"      - {col.name} ({col.dtype})")


def demonstrate_transform(dataset_path: Path, output_path: Path, transform_type: str):
    """Demonstrate running data transforms."""
    logger.info(f"--- Running {transform_type} Transform ---")
    
    fastdfs.run_transform(
        dataset_path=dataset_path,
        output_path=output_path,
        transform_type=transform_type
    )
    
    logger.info(f"Transform completed! Output saved to: {output_path}")
    
    # Show output structure
    if output_path.exists():
        logger.info("Output directory contents:")
        for item in output_path.iterdir():
            logger.info(f"  {item.name}")


def demonstrate_dfs(dataset_path: Path, output_path: Path, max_depth: int, engine: str):
    """Demonstrate running Deep Feature Synthesis."""
    logger.info(f"--- Running DFS (depth={max_depth}, engine={engine}) ---")
    
    fastdfs.run_dfs(
        dataset_path=dataset_path,
        output_path=output_path,
        max_depth=max_depth,
        engine=engine,
        use_cutoff_time=False
    )
    
    logger.info(f"DFS completed! Output saved to: {output_path}")
    
    # Show output structure
    if output_path.exists():
        logger.info("Output directory contents:")
        for item in output_path.iterdir():
            logger.info(f"  {item.name}")


def demonstrate_full_pipeline(dataset_path: Path, output_path: Path, max_depth: int, engine: str):
    """Demonstrate running the complete DFS pipeline."""
    logger.info(f"--- Running Full Pipeline (depth={max_depth}, engine={engine}) ---")
    
    fastdfs.run_full_pipeline(
        dataset_path=dataset_path,
        output_path=output_path,
        max_depth=max_depth,
        engine=engine,
        use_cutoff_time=False
    )
    
    logger.info(f"Full pipeline completed! Final output saved to: {output_path}")
    
    # Show output structure
    if output_path.exists():
        logger.info("Final output directory contents:")
        for item in output_path.iterdir():
            logger.info(f"  {item.name}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
