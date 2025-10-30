#!/usr/bin/env python3
"""
Create minimal example 1: Direct Features (Depth 0)

This script creates a minimal RDB for testing direct feature extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Create directory structure
example_dir = Path(__file__).parent
data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)

# Create minimal user table
# 3 users with different attributes
user_data = {
    'user_id': np.array([1, 2, 3], dtype=object),
    'age': np.array([25.0, 30.0, 35.0], dtype=np.float32),
    'city': np.array(['NYC', 'SF', 'LA'], dtype=object)
}

# Save as npz (format used by fastdfs)
np.savez(
    data_dir / "user.npz",
    **user_data
)

# Create metadata.yaml
metadata = {
    'dataset_name': 'example1_direct_features',
    'tables': [
        {
            'name': 'user',
            'source': 'data/user.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'user_id', 'dtype': 'primary_key'},
                {'name': 'age', 'dtype': 'float'},
                {'name': 'city', 'dtype': 'category'}
            ]
        }
    ]
}

with open(example_dir / "metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Create target table example
target_df = pd.DataFrame({
    'user_id': [1, 2, 3, 1],  # Some users appear multiple times
    'target_label': [0, 1, 0, 1]
})

target_df.to_csv(example_dir / "target_table.csv", index=False)

print(f"Created Example 1 in: {example_dir}")
print(f"\nUser table data:")
for key, value in user_data.items():
    print(f"  {key}: {value}")
print(f"\nTarget table shape: {target_df.shape}")
print(f"Target table:\n{target_df}")
print(f"\nKey mappings for DFS:")
print("  user_id -> user.user_id")

# Generate the golden DataFrame as in example output and save to ../data/golden.npz
golden_df = pd.DataFrame({
    'user_id': [1, 2, 3, 1],
    'target_label': [0, 1, 0, 1],
    'user.age': [25.0, 30.0, 35.0, 25.0]
})

golden_npz_path = example_dir / "data" / "golden_depth_1.npz"
golden_npz_path.parent.mkdir(exist_ok=True, parents=True)

np.savez(
    golden_npz_path,
    **{col: golden_df[col].to_numpy() for col in golden_df.columns}
)

print(f"\nGolden table saved to: {golden_npz_path}")
print(f"\nGolden table contents:")
print(golden_df)
