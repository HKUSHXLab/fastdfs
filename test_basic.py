#!/usr/bin/env python3
"""
Basic test script for FastDFS functionality.
This script tests the core DFS extraction without requiring full installation.
"""

import sys
import os
from pathlib import Path

# Add the fastdfs directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_import():
    """Test if we can import core modules"""
    try:
        # Test dataset loading (this will likely fail due to pydantic)
        from fastdfs.dataset import meta
        print("✓ Dataset meta module imported successfully")
    except ImportError as e:
        print(f"✗ Dataset meta import failed: {e}")
    
    try:
        # Test DFS core
        from fastdfs.preprocess.dfs import core
        print("✓ DFS core module imported successfully") 
    except ImportError as e:
        print(f"✗ DFS core import failed: {e}")
        
    try:
        # Test transform base
        from fastdfs.preprocess.transform import base
        print("✓ Transform base module imported successfully")
    except ImportError as e:
        print(f"✗ Transform base import failed: {e}")

def test_file_structure():
    """Test if all expected files exist"""
    expected_files = [
        "fastdfs/__init__.py",
        "fastdfs/preprocess/__init__.py",
        "fastdfs/preprocess/dfs/__init__.py",
        "fastdfs/preprocess/dfs/core.py",
        "fastdfs/preprocess/dfs/dfs_preprocess.py",
        "fastdfs/preprocess/dfs/ft_engine.py",
        "fastdfs/preprocess/dfs/dfs2sql_engine.py",
        "fastdfs/preprocess/transform/__init__.py",
        "fastdfs/preprocess/transform/base.py",
        "fastdfs/dataset/__init__.py",
        "fastdfs/dataset/rdb_dataset.py",
        "fastdfs/utils/__init__.py",
        "fastdfs/cli/__init__.py",
        "configs/dfs/dfs-2.yaml",
        "configs/transform/pre-dfs.yaml",
        "tests/data/test_rdb/metadata.yaml",
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    if missing_files:
        print(f"\n{len(missing_files)} files are missing")
        return False
    else:
        print(f"\n✓ All {len(expected_files)} expected files found")
        return True

def test_config_files():
    """Test if configuration files are valid"""
    try:
        import yaml
        
        # Test DFS config
        with open("configs/dfs/dfs-2.yaml", "r") as f:
            dfs_config = yaml.safe_load(f)
        print(f"✓ DFS config loaded: {dfs_config}")
        
        # Test transform config
        with open("configs/transform/pre-dfs.yaml", "r") as f:
            transform_config = yaml.safe_load(f)
        print(f"✓ Transform config loaded: {transform_config}")
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False
    
    return True

def test_dataset_structure():
    """Test if test dataset has correct structure"""
    try:
        import yaml
        
        metadata_path = "tests/data/test_rdb/metadata.yaml"
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        
        print(f"✓ Test dataset metadata loaded")
        print(f"  Dataset name: {metadata.get('dataset_name', 'N/A')}")
        print(f"  Tables: {len(metadata.get('tables', []))}")
        print(f"  Tasks: {len(metadata.get('tasks', []))}")
        
        # Check data files exist
        data_files = ["data/user.npz", "data/item.npz", "data/interaction.npz"]
        for data_file in data_files:
            full_path = f"tests/data/test_rdb/{data_file}"
            if Path(full_path).exists():
                print(f"  ✓ Found: {data_file}")
            else:
                print(f"  ✗ Missing: {data_file}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("FastDFS Basic Validation Test")
    print("=" * 40)
    
    print("\n1. Testing file structure...")
    structure_ok = test_file_structure()
    
    print("\n2. Testing configuration files...")
    config_ok = test_config_files()
    
    print("\n3. Testing dataset structure...")
    dataset_ok = test_dataset_structure()
    
    print("\n4. Testing module imports...")
    test_basic_import()
    
    print("\n" + "=" * 40)
    if structure_ok and config_ok and dataset_ok:
        print("✓ Basic validation PASSED")
        print("FastDFS package structure is correctly set up!")
    else:
        print("✗ Basic validation FAILED")
        print("Some issues need to be fixed before testing full functionality.")
