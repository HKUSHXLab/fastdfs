#!/usr/bin/env python3
"""
Test script for FastDFS enhanced API
"""

import sys
import logging
from pathlib import Path
import tempfile
import shutil

# Add the fastdfs package to the path
sys.path.insert(0, '/home/ubuntu/git-repo/tab2graph/fastdfs')

import fastdfs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_load_dataset():
    """Test loading a dataset"""
    logger.info("=== Test 1: Loading Dataset ===")
    
    dataset_path = "/home/ubuntu/git-repo/tab2graph/datasets_synthetic/test_rdb"
    
    try:
        dataset = fastdfs.load_dataset(dataset_path)
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dataset name: {dataset.metadata.dataset_name}")
        logger.info(f"Number of tables: {len(dataset.metadata.tables)}")
        
        for table in dataset.metadata.tables:
            logger.info(f"  Table: {table.name} with {len(table.columns)} columns")
            
        return True
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

def test_transform():
    """Test running transforms"""
    logger.info("=== Test 2: Transform Processing ===")
    
    dataset_path = "/home/ubuntu/git-repo/tab2graph/datasets_synthetic/test_rdb"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "transformed"
        
        try:
            fastdfs.run_transform(
                dataset_path=dataset_path,
                output_path=output_path,
                transform_type="pre-dfs"
            )
            
            # Check if output was created
            if output_path.exists():
                logger.info(f"Transform completed successfully! Output at {output_path}")
                return True
            else:
                logger.error("Transform output directory not created")
                return False
                
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_dfs():
    """Test running DFS"""
    logger.info("=== Test 3: DFS Processing ===")
    
    dataset_path = "/home/ubuntu/git-repo/tab2graph/datasets_synthetic/test_rdb"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "dfs_output"
        
        try:
            fastdfs.run_dfs(
                dataset_path=dataset_path,
                output_path=output_path,
                max_depth=2,
                engine="featuretools",
                use_cutoff_time=False
            )
            
            # Check if output was created
            if output_path.exists():
                logger.info(f"DFS completed successfully! Output at {output_path}")
                return True
            else:
                logger.error("DFS output directory not created")
                return False
                
        except Exception as e:
            logger.error(f"DFS failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_full_pipeline():
    """Test the full pipeline"""
    logger.info("=== Test 4: Full Pipeline ===")
    
    dataset_path = "/home/ubuntu/git-repo/tab2graph/datasets_synthetic/test_rdb"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "pipeline_output"
        
        try:
            fastdfs.run_full_pipeline(
                dataset_path=dataset_path,
                output_path=output_path,
                max_depth=1,  # Use smaller depth for faster testing
                engine="featuretools",
                use_cutoff_time=False
            )
            
            # Check if output was created
            if output_path.exists():
                logger.info(f"Full pipeline completed successfully! Output at {output_path}")
                return True
            else:
                logger.error("Full pipeline output directory not created")
                return False
                
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests"""
    logger.info("Starting FastDFS API tests...")
    
    results = {
        "load_dataset": test_load_dataset(),
        "transform": test_transform(),
        "dfs": test_dfs(),
        "full_pipeline": test_full_pipeline()
    }
    
    logger.info("=== Test Results ===")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
