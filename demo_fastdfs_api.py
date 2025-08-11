#!/usr/bin/env python3
"""
Demo script showing FastDFS enhanced API usage
"""

import sys
import logging
from pathlib import Path
import tempfile

# Add the fastdfs package to the path
sys.path.insert(0, '/home/ubuntu/git-repo/tab2graph/fastdfs')

import fastdfs

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_enhanced_api():
    """Demonstrate the enhanced FastDFS API"""
    
    dataset_path = "/home/ubuntu/git-repo/tab2graph/datasets_synthetic/test_rdb"
    
    print("🚀 FastDFS Enhanced API Demo")
    print("=" * 50)
    
    # 1. Load and inspect dataset
    print("\n📊 Loading dataset...")
    dataset = fastdfs.load_dataset(dataset_path)
    print(f"✓ Dataset: {dataset.metadata.dataset_name}")
    print(f"✓ Tables: {len(dataset.metadata.tables)}")
    for table in dataset.metadata.tables:
        print(f"  - {table.name}: {len(table.columns)} columns")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 2. Test different engines
        print("\n🔧 Testing DFS with different engines...")
        
        # Test featuretools engine
        ft_output = temp_path / "featuretools_output"
        print("  • Running with Featuretools engine...")
        fastdfs.run_dfs(
            dataset_path=dataset_path,
            output_path=ft_output,
            max_depth=2,
            engine="featuretools"
        )
        print(f"    ✓ Featuretools DFS completed")
        
        # Test SQL engine
        sql_output = temp_path / "sql_output"
        print("  • Running with DFS2SQL engine...")
        fastdfs.run_dfs(
            dataset_path=dataset_path,
            output_path=sql_output,
            max_depth=2,
            engine="dfs2sql"
        )
        print(f"    ✓ DFS2SQL completed")
        
        # 3. Test transforms
        print("\n🔄 Testing data transforms...")
        
        for transform_type in ["pre-dfs", "post-dfs"]:
            output_path = temp_path / f"transform_{transform_type.replace('-', '_')}"
            print(f"  • Running {transform_type} transforms...")
            fastdfs.run_transform(
                dataset_path=dataset_path,
                output_path=output_path,
                transform_type=transform_type
            )
            print(f"    ✓ {transform_type} transforms completed")
        
        # 4. Test full pipeline
        print("\n🎯 Testing full pipeline...")
        pipeline_output = temp_path / "full_pipeline"
        fastdfs.run_full_pipeline(
            dataset_path=dataset_path,
            output_path=pipeline_output,
            max_depth=1,  # Use depth 1 for speed
            engine="featuretools"
        )
        print(f"    ✓ Full pipeline completed")
        
        # 5. Show output structure  
        print("\n📁 Output structures:")
        for output_dir in [ft_output, sql_output, pipeline_output]:
            if output_dir.exists():
                files = list(output_dir.rglob("*"))
                print(f"  • {output_dir.name}: {len(files)} files")
                if len(files) <= 10:
                    for file in sorted(files):
                        print(f"    - {file.relative_to(output_dir)}")
    
    print("\n✅ FastDFS Enhanced API Demo completed successfully!")
    
    # 6. Show import examples
    print("\n💡 Usage Examples:")
    print("""
# Basic DFS
import fastdfs
fastdfs.run_dfs('dataset/', 'output/', max_depth=2, engine='featuretools')

# Full pipeline  
fastdfs.run_full_pipeline('dataset/', 'output/', max_depth=2)

# Load and inspect dataset
dataset = fastdfs.load_dataset('dataset/')
print(f"Dataset: {dataset.metadata.dataset_name}")
""")

if __name__ == "__main__":
    demo_enhanced_api()
