# FastDFS Examples

This directory contains examples demonstrating how to use FastDFS for automated feature engineering using Deep Feature Synthesis (DFS).

## Available Examples

### 1. Python API Example (`python_api_example.py`)

Demonstrates how to use the FastDFS Python API for various operations:

- **Loading datasets**: Inspect dataset structure and metadata
- **Running transforms**: Apply pre-processing, post-processing, or standalone transforms
- **Running DFS**: Perform Deep Feature Synthesis with different engines and depths
- **Full pipeline**: Complete workflow from raw data to engineered features

#### Usage

```bash
# Basic usage with default test dataset
python examples/python_api_example.py

# Use custom dataset
python examples/python_api_example.py --dataset-path /path/to/your/dataset

# Configure DFS parameters
python examples/python_api_example.py --max-depth 3 --engine dfs2sql

# Run specific operations
python examples/python_api_example.py --operation load
python examples/python_api_example.py --operation transform --transform-type pre-dfs
python examples/python_api_example.py --operation dfs --max-depth 2
python examples/python_api_example.py --operation full-pipeline
```

#### Available Options

- `--dataset-path`: Path to dataset directory (default: `tests/data/test_rdb`)
- `--output-path`: Output directory (default: temporary directory)
- `--max-depth`: Maximum DFS depth (default: 2)
- `--engine`: DFS engine - "featuretools" or "dfs2sql" (default: "featuretools")
- `--operation`: Operation to perform - "load", "transform", "dfs", "full-pipeline"
- `--transform-type`: Transform type - "pre-dfs", "post-dfs", "single"

### 2. CLI Example (`cli_example.py`)

Demonstrates how to use the FastDFS command-line interface:

- **Transform commands**: Pre-DFS, post-DFS, and single transforms
- **DFS commands**: Different engines, depths, and configurations
- **Configuration files**: Using different YAML configuration files
- **Pipeline workflows**: Step-by-step feature engineering pipeline

#### Usage

```bash
# Show CLI commands without executing them
python examples/cli_example.py

# Execute CLI commands with default dataset
python examples/cli_example.py --execute

# Use custom dataset
python examples/cli_example.py --dataset-path /path/to/your/dataset --execute

# Specify output directory
python examples/cli_example.py --output-path /path/to/output --execute
```

#### Available Options

- `--dataset-path`: Path to dataset directory (default: `tests/data/test_rdb`)
- `--output-path`: Output directory (default: temporary directory)
- `--execute`: Actually run the commands (default: just show them)

## Test Dataset

Both examples use the test dataset located at `tests/data/test_rdb/` by default. This dataset contains:

- **User table**: User entities with features
- **Item table**: Item entities with features  
- **Interaction table**: User-item interactions with timestamps

The dataset is in the RDB (Relational Database) format with:
- `metadata.yaml`: Dataset schema and table definitions
- `data/`: Directory containing table data files
- `linkpred/`: Task-specific data for link prediction

## Configuration Files

FastDFS uses YAML configuration files located in the `configs/` directory:

### DFS Configurations (`configs/dfs/`)

- `dfs-1.yaml`: DFS with depth 1 using featuretools
- `dfs-2.yaml`: DFS with depth 2 using featuretools
- `dfs-3.yaml`: DFS with depth 3 using featuretools
- `dfs-1-sql.yaml`: DFS with depth 1 using SQL engine
- `dfs-2-sql.yaml`: DFS with depth 2 using SQL engine
- `dfs-3-sql.yaml`: DFS with depth 3 using SQL engine

### Transform Configurations (`configs/transform/`)

- `pre-dfs.yaml`: Pre-processing transforms applied before DFS
- `post-dfs.yaml`: Post-processing transforms applied after DFS
- `single.yaml`: Standalone transforms without DFS

## Common Workflows

### 1. Complete Pipeline (Python API)

```python
import fastdfs

# Run complete pipeline: pre-transform → DFS → post-transform
fastdfs.run_full_pipeline(
    dataset_path="tests/data/test_rdb",
    output_path="output/final",
    max_depth=2,
    engine="featuretools"
)
```

### 2. Complete Pipeline (CLI)

```bash
# Step 1: Pre-DFS transforms
fastdfs preprocess tests/data/test_rdb transform pre_dfs -c configs/transform/pre-dfs.yaml

# Step 2: Run DFS
fastdfs preprocess pre_dfs dfs dfs_output -c configs/dfs/dfs-2.yaml

# Step 3: Post-DFS transforms
fastdfs preprocess dfs_output transform final -c configs/transform/post-dfs.yaml
```

### 3. Quick DFS Only

```bash
# Python API
fastdfs.run_dfs("tests/data/test_rdb", "dfs_output", max_depth=2, engine="featuretools")

# CLI
fastdfs preprocess tests/data/test_rdb dfs dfs_output -c configs/dfs/dfs-2.yaml
```

### 4. Custom Configuration

```python
# Use custom config file
fastdfs.run_dfs(
    dataset_path="tests/data/test_rdb",
    output_path="output",
    config_path="my_custom_config.yaml"
)
```

## Output Structure

FastDFS generates output datasets in the same RDB format as input:

```
output_directory/
├── metadata.yaml          # Updated metadata with new features
├── data/                  # Directory with processed table data
│   ├── table1.npz        # Processed data files
│   ├── table2.npz
│   └── ...
└── linkpred/             # Task-specific data (if applicable)
```

## Requirements

Make sure FastDFS is installed before running the examples:

```bash
cd fastdfs
pip install -e .
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure FastDFS is installed with `pip install -e .`
2. **Dataset not found**: Check that the dataset path exists and contains `metadata.yaml`
3. **Config file errors**: Verify configuration file paths and YAML syntax
4. **Memory issues**: Try reducing DFS depth or using the SQL engine for large datasets

### Getting Help

```bash
# FastDFS help
fastdfs --help

# Preprocess command help
fastdfs preprocess --help

# Python API documentation
python -c "import fastdfs; help(fastdfs)"
```

## Next Steps

- Try the examples with your own datasets
- Experiment with different DFS depths and engines
- Create custom configuration files for your use cases
- Explore the generated features and their impact on downstream tasks
