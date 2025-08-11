# FastDFS - Deep Feature Synthesis for Tabular Data

FastDFS is a standalone package for automated feature engineering using Deep Feature Synthesis (DFS). Extracted from the tab2graph project, it provides focused functionality for relational feature generation.

## Quick Start

### Installation

```bash
cd fastdfs
pip install -e .
```

### Usage

#### Command Line Interface

```bash
# Run pre-DFS transforms
fastdfs preprocess path/to/dataset transform output_dataset -c configs/transform/pre-dfs.yaml

# Run DFS with depth 2
fastdfs preprocess dataset dfs output_dataset -c configs/dfs/dfs-2.yaml

# Run post-DFS transforms  
fastdfs preprocess dataset transform output_dataset -c configs/transform/post-dfs.yaml
```

#### Python API

```python
import fastdfs

# Load dataset
dataset = fastdfs.load_rdb_data("path/to/dataset")

# Configure and run DFS
from fastdfs.preprocess.dfs import DFSPreprocess, DFSPreprocessConfig
from fastdfs.preprocess.dfs.core import DFSConfig

config = DFSPreprocessConfig(dfs=DFSConfig(max_depth=2, engine="featuretools"))
processor = DFSPreprocess(config)
processor.run(dataset, "output_path", device_info)
```

## Features

- **Multiple DFS Engines**: Featuretools and high-performance SQL-based engines
- **Flexible Configuration**: YAML-based configuration system
- **Transform Pipeline**: Pre and post-DFS data transformations
- **Custom Primitives**: Support for custom aggregation primitives
- **Minimal Dependencies**: Focused on core DFS functionality

## Configuration

Configuration files are located in the `configs/` directory:

- `configs/dfs/`: DFS engine configurations
- `configs/transform/`: Data transformation configurations

## Testing

The package includes a test dataset in `tests/data/test_rdb/` for validation.

## Migration from tab2graph

FastDFS maintains API compatibility with the tab2graph preprocessing commands:

```bash
# Before (tab2graph)
t2g preprocess dataset dfs output -c configs/dfs/dfs-2.yaml

# After (fastdfs)  
fastdfs preprocess dataset dfs output -c configs/dfs/dfs-2.yaml
```
