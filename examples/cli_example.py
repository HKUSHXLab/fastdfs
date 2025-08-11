#!/usr/bin/env python3
"""
FastDFS CLI Example

This example demonstrates how to use the FastDFS command-line interface for
automated feature engineering using Deep Feature Synthesis (DFS).

This script provides examples of various CLI commands and can be run to
execute them step by step.

Usage:
    python cli_example.py [--dataset-path PATH] [--output-path PATH] [--execute]

Examples:
    # Show CLI commands without executing
    python cli_example.py
    
    # Execute CLI commands with default dataset
    python cli_example.py --execute
    
    # Use custom dataset path
    python cli_example.py --dataset-path /path/to/your/dataset --execute
"""

import argparse
from loguru import logger
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastdfs.utils.logging_config import configure_logging

# Configure logging for the example
configure_logging(level="INFO")


def run_command(command: str, description: str, execute: bool = False):
    """Run a CLI command or just display it."""
    logger.info(f"\n--- {description} ---")
    logger.info(f"Command: {command}")
    
    if execute:
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Command executed successfully!")
            if result.stdout:
                logger.info(f"Output:\n{result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            if e.stderr:
                logger.error(f"Error output:\n{e.stderr}")
            return False
    else:
        logger.info("(Command not executed - use --execute flag to run)")
        return True


def main():
    """Main function to demonstrate FastDFS CLI usage."""
    parser = argparse.ArgumentParser(
        description="FastDFS CLI Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Show commands
  %(prog)s --execute                          # Execute with default dataset
  %(prog)s --dataset-path /path/to/data --execute
        """
    )
    
    # Default to test dataset relative to project root
    project_root = Path(__file__).parent.parent
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
        "--execute",
        action="store_true",
        help="Actually execute the commands (default: just show them)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not args.dataset_path.exists():
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        return False
    
    # Set up output paths
    if args.output_path is None:
        temp_dir = tempfile.mkdtemp(prefix="fastdfs_cli_example_")
        base_output = Path(temp_dir)
        cleanup_temp = True
        logger.info(f"Using temporary output directory: {base_output}")
    else:
        base_output = args.output_path
        cleanup_temp = False
    
    # Create output subdirectories
    pre_dfs_output = base_output / "pre_dfs"
    dfs_output = base_output / "dfs"
    post_dfs_output = base_output / "post_dfs"
    single_transform_output = base_output / "single_transform"
    
    try:
        logger.info("=== FastDFS CLI Examples ===")
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Base output path: {base_output}")
        
        if not args.execute:
            logger.info("\nNOTE: Commands will be displayed but not executed.")
            logger.info("Use --execute flag to actually run the commands.")
        
        success = True
        
        # Example 1: Pre-DFS transforms
        command = f"fastdfs preprocess {args.dataset_path} transform {pre_dfs_output} -c {project_root}/configs/transform/pre-dfs.yaml"
        success &= run_command(command, "Pre-DFS Transform", args.execute)
        
        # Example 2: DFS with depth 2 using featuretools
        dfs_input = pre_dfs_output if args.execute and pre_dfs_output.exists() else args.dataset_path
        command = f"fastdfs preprocess {dfs_input} dfs {dfs_output} -c {project_root}/configs/dfs/dfs-2.yaml"
        success &= run_command(command, "DFS with Depth 2 (Featuretools)", args.execute)
        
        # Example 3: Post-DFS transforms
        post_dfs_input = dfs_output if args.execute and dfs_output.exists() else args.dataset_path
        command = f"fastdfs preprocess {post_dfs_input} transform {post_dfs_output} -c {project_root}/configs/transform/post-dfs.yaml"
        success &= run_command(command, "Post-DFS Transform", args.execute)
        
        # Example 4: Single transform (standalone)
        command = f"fastdfs preprocess {args.dataset_path} transform {single_transform_output} -c {project_root}/configs/transform/single.yaml"
        success &= run_command(command, "Single Transform", args.execute)
        
        # Example 5: DFS with SQL engine
        dfs_sql_output = base_output / "dfs_sql"
        command = f"fastdfs preprocess {args.dataset_path} dfs {dfs_sql_output} -c {project_root}/configs/dfs/dfs-2-sql.yaml"
        success &= run_command(command, "DFS with SQL Engine", args.execute)
        
        # Example 6: DFS with depth 3
        dfs_depth3_output = base_output / "dfs_depth3"
        command = f"fastdfs preprocess {args.dataset_path} dfs {dfs_depth3_output} -c {project_root}/configs/dfs/dfs-3.yaml"
        success &= run_command(command, "DFS with Depth 3", args.execute)
        
        # Show additional CLI information
        show_additional_info(project_root, args.execute)
        
        if success:
            logger.info("\n=== CLI Examples completed successfully! ===")
        else:
            logger.warning("\n=== Some CLI commands failed ===")
            
        return success
        
    except Exception as e:
        logger.error(f"CLI examples failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory if used
        if cleanup_temp and Path(temp_dir).exists():
            if args.execute:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            else:
                logger.info(f"Temporary directory (not cleaned): {temp_dir}")


def show_additional_info(project_root: Path, execute: bool):
    """Show additional CLI information and available configurations."""
    logger.info("\n--- Additional CLI Information ---")
    
    # Show available config files
    logger.info("\nAvailable DFS configurations:")
    dfs_configs = project_root / "configs" / "dfs"
    if dfs_configs.exists():
        for config_file in dfs_configs.glob("*.yaml"):
            logger.info(f"  {config_file.name}")
    
    logger.info("\nAvailable transform configurations:")
    transform_configs = project_root / "configs" / "transform"
    if transform_configs.exists():
        for config_file in transform_configs.glob("*.yaml"):
            logger.info(f"  {config_file.name}")
    
    # Show help command
    logger.info("\nTo see all available commands and options:")
    help_command = "fastdfs --help"
    run_command(help_command, "FastDFS Help", execute)
    
    logger.info("\nTo see preprocess command options:")
    preprocess_help_command = "fastdfs preprocess --help"
    run_command(preprocess_help_command, "Preprocess Help", execute)
    
    # Show common usage patterns
    logger.info("\n--- Common Usage Patterns ---")
    logger.info("1. Complete pipeline using CLI:")
    logger.info("   fastdfs preprocess data transform pre_dfs -c configs/transform/pre-dfs.yaml")
    logger.info("   fastdfs preprocess pre_dfs dfs dfs_output -c configs/dfs/dfs-2.yaml")
    logger.info("   fastdfs preprocess dfs_output transform final -c configs/transform/post-dfs.yaml")
    
    logger.info("\n2. Quick DFS without transforms:")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-2.yaml")
    
    logger.info("\n3. Different DFS engines:")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-2.yaml      # Featuretools")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-2-sql.yaml  # SQL engine")
    
    logger.info("\n4. Different depths:")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-1.yaml  # Depth 1")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-2.yaml  # Depth 2")
    logger.info("   fastdfs preprocess data dfs output -c configs/dfs/dfs-3.yaml  # Depth 3")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
