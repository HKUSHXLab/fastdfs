#!/usr/bin/env python3
"""
Create minimal example 2: Single-Level Aggregation (Depth 1)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Create directory structure
example_dir = Path(__file__).parent
data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)

# Create user table
user_data = {
    'user_id': np.array([1, 2, 3], dtype=object),
    'age': np.array([25.0, 30.0, 35.0], dtype=np.float32),
}

# Create transaction table (child of user)
# User 1: 3 transactions [10, 20, 30] in categories [A, A, B]
# User 2: 2 transactions [15, 25] in categories [A, B]
# User 3: 1 transaction [50] in category [B]
transaction_data = {
    'transaction_id': np.array([1, 2, 3, 4, 5, 6], dtype=object),
    'user_id': np.array([1, 1, 1, 2, 2, 3], dtype=object),
    'amount': np.array([10.0, 20.0, 30.0, 15.0, 25.0, 50.0], dtype=np.float32),
    'category': np.array(['A', 'A', 'B', 'A', 'B', 'B'], dtype=object)
}

# Save as npz
np.savez(data_dir / "user.npz", **user_data)
np.savez(data_dir / "transaction.npz", **transaction_data)

# Create metadata.yaml
metadata = {
    'dataset_name': 'example2_single_aggregation',
    'tables': [
        {
            'name': 'user',
            'source': 'data/user.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'user_id', 'dtype': 'primary_key'},
                {'name': 'age', 'dtype': 'float'}
            ]
        },
        {
            'name': 'transaction',
            'source': 'data/transaction.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'transaction_id', 'dtype': 'primary_key'},
                {'name': 'user_id', 'dtype': 'foreign_key', 'link_to': 'user.user_id'},
                {'name': 'amount', 'dtype': 'float'},
                {'name': 'category', 'dtype': 'category'}
            ]
        }
    ]
}

with open(example_dir / "metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Create target table
target_df = pd.DataFrame({
    'user_id': [1, 2, 3, 1],  # User 1 appears twice
    'target_label': [0, 1, 0, 1]
})

target_df.to_csv(example_dir / "target_table.csv", index=False)

print(f"Created Example 2 in: {example_dir}")
print(f"\nUser table: {len(user_data['user_id'])} users")
print(f"Transaction table: {len(transaction_data['transaction_id'])} transactions")
print(f"\nTransaction breakdown by user:")
for user_id in [1, 2, 3]:
    mask = transaction_data['user_id'] == user_id
    amounts = transaction_data['amount'][mask]
    categories = transaction_data['category'][mask]
    print(f"  User {user_id}: {len(amounts)} transactions")
    print(f"    Amounts: {amounts.tolist()}")
    print(f"    Categories: {categories.tolist()}")
    print(f"    Expected COUNT: {len(amounts)}")
    print(f"    Expected MEAN: {amounts.mean():.1f}")
    print(f"    Expected MAX: {amounts.max():.1f}")
    print(f"    Expected MIN: {amounts.min():.1f}")
    print()

print(f"Target table shape: {target_df.shape}")
print(f"Target table:\n{target_df}")


# Construct the data for golden_depth_2, based on the table at file_context_0
golden_data = {
    "user_id": np.array([1, 2, 3, 1], dtype=np.int64),
    "target_label": np.array([0, 1, 0, 1], dtype=np.int64),
    "user.age": np.array([25.0, 30.0, 35.0, 25.0], dtype=float),
    "user.MEAN(transaction.amount)": np.array([20.0, 20.0, 50.0, 20.0], dtype=float),
    "user.COUNT(transaction)": np.array([3, 2, 1, 3], dtype=np.int64),
    "user.MIN(transaction.amount)": np.array([10.0, 15.0, 50.0, 10.0], dtype=float),
    "user.MODE(transaction.category)": np.array(['A', 'A', 'B', 'A'], dtype=object),
    "user.MAX(transaction.amount)": np.array([30.0, 25.0, 50.0, 30.0], dtype=float),
}

data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)
np.savez_compressed(data_dir / "golden_depth_2.npz", **golden_data)
print(f"Saved golden_depth_2.npz to {data_dir}")
