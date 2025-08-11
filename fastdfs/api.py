"""
Enhanced API for FastDFS - Simplified interface for common DFS workflows
"""

from pathlib import Path
from typing import Optional, Union
import logging

from .dataset import load_rdb_data, DBBRDBDataset
from .preprocess.dfs import DFSPreprocess, DFSPreprocessConfig
from .preprocess.transform_preprocess import RDBTransformPreprocess, RDBTransformPreprocessConfig
from .preprocess.dfs.core import DFSConfig
from .utils.device import get_device_info
from .utils import yaml_utils

logger = logging.getLogger(__name__)

def run_dfs(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path],
    max_depth: int = 2,
    engine: str = "featuretools",
    config_path: Optional[Union[str, Path]] = None,
    use_cutoff_time: bool = False
) -> None:
    """Run DFS on a dataset with simplified interface.
    
    Args:
        dataset_path: Path to input dataset directory
        output_path: Path for output dataset
        max_depth: Maximum depth for DFS (default: 2)
        engine: DFS engine to use - "featuretools" or "dfs2sql" (default: "featuretools") 
        config_path: Optional path to custom DFS config file
        use_cutoff_time: Whether to use cutoff time for temporal features
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_rdb_data(str(dataset_path))
    
    if config_path:
        logger.info(f"Loading DFS config from {config_path}")
        config = yaml_utils.load_pyd(DFSPreprocessConfig, config_path)
    else:
        logger.info(f"Using default DFS config with depth={max_depth}, engine={engine}")
        dfs_config = DFSConfig(
            max_depth=max_depth,
            engine=engine,
            use_cutoff_time=use_cutoff_time
        )
        config = DFSPreprocessConfig(dfs=dfs_config)
    
    logger.info("Running DFS preprocessing...")
    processor = DFSPreprocess(config)
    device = get_device_info()
    processor.run(dataset, output_path, device)
    
    logger.info(f"DFS completed! Output saved to {output_path}")


def run_transform(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path], 
    transform_type: str = "pre-dfs",
    config_path: Optional[Union[str, Path]] = None
) -> None:
    """Run data transforms with simplified interface.
    
    Args:
        dataset_path: Path to input dataset directory
        output_path: Path for output dataset
        transform_type: Type of transforms - "pre-dfs", "post-dfs", or "single"
        config_path: Optional path to custom transform config file
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_rdb_data(str(dataset_path))
    
    if config_path:
        logger.info(f"Loading transform config from {config_path}")
        config = yaml_utils.load_pyd(RDBTransformPreprocessConfig, config_path)
    else:
        # Use default configs based on transform type
        fastdfs_root = Path(__file__).parent.parent  # Go up from fastdfs/fastdfs/ to fastdfs/
        default_configs = {
            "pre-dfs": fastdfs_root / "configs" / "transform" / "pre-dfs.yaml",
            "post-dfs": fastdfs_root / "configs" / "transform" / "post-dfs.yaml", 
            "single": fastdfs_root / "configs" / "transform" / "single.yaml"
        }
        
        if transform_type not in default_configs:
            raise ValueError(f"Unknown transform_type: {transform_type}. Must be one of {list(default_configs.keys())}")
        
        config_file = default_configs[transform_type]
        logger.info(f"Using default {transform_type} config from {config_file}")
        config = yaml_utils.load_pyd(RDBTransformPreprocessConfig, config_file)
    
    logger.info(f"Running {transform_type} transforms...")
    processor = RDBTransformPreprocess(config)
    device = get_device_info()
    processor.run(dataset, output_path, device)
    
    logger.info(f"Transform completed! Output saved to {output_path}")


def run_full_pipeline(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path],
    max_depth: int = 2,
    engine: str = "featuretools",
    use_cutoff_time: bool = False
) -> None:
    """Run complete DFS pipeline: pre-transform → DFS → post-transform.
    
    Args:
        dataset_path: Path to input dataset directory
        output_path: Path for final output dataset
        max_depth: Maximum depth for DFS (default: 2)
        engine: DFS engine to use - "featuretools" or "dfs2sql" (default: "featuretools")
        use_cutoff_time: Whether to use cutoff time for temporal features
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Create intermediate directories
    temp_dir = output_path.parent / f"{output_path.name}_temp"
    pre_dfs_path = temp_dir / "pre_dfs"
    dfs_path = temp_dir / "dfs" 
    
    logger.info("Starting full DFS pipeline...")
    
    try:
        # Step 1: Pre-DFS transforms
        logger.info("Step 1/3: Running pre-DFS transforms...")
        run_transform(dataset_path, pre_dfs_path, "pre-dfs")
        
        # Step 2: DFS
        logger.info("Step 2/3: Running DFS...")
        run_dfs(pre_dfs_path, dfs_path, max_depth, engine, use_cutoff_time=use_cutoff_time)
        
        # Step 3: Post-DFS transforms  
        logger.info("Step 3/3: Running post-DFS transforms...")
        run_transform(dfs_path, output_path, "post-dfs")
        
        logger.info(f"Full pipeline completed! Final output saved to {output_path}")
        
    finally:
        # Clean up temporary files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory {temp_dir}")


def load_dataset(dataset_path: Union[str, Path]) -> DBBRDBDataset:
    """Load a dataset for inspection or custom processing.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        DBBRDBDataset: The loaded dataset
    """
    return load_rdb_data(str(dataset_path))


def save_dataset(dataset: DBBRDBDataset, output_path: Union[str, Path]) -> None:
    """Save a dataset (placeholder - would need implementation).
    
    Args:
        dataset: Dataset to save
        output_path: Output directory path
    """
    # This would need to be implemented based on dataset creator functionality
    raise NotImplementedError("Dataset saving not yet implemented")
