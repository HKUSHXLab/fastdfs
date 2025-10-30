#!/usr/bin/env python3
"""
Create minimal example 4: Cutoff Time Handling
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
    'user_id': np.array([1, 2], dtype=object),
    'age': np.array([25.0, 30.0], dtype=np.float32),
}

# Create transaction table with timestamps
# User 1: 3 transactions at different times
# User 2: 2 transactions at different times
transaction_data = {
    'transaction_id': np.array([1, 2, 3, 4, 5], dtype=object),
    'user_id': np.array([1, 1, 1, 2, 2], dtype=object),
    'amount': np.array([10.0, 20.0, 30.0, 15.0, 25.0], dtype=np.float32),
    'timestamp': np.array([
        '2024-01-01T10:00:00',
        '2024-01-02T11:00:00',
        '2024-01-03T12:00:00',
        '2024-01-01T14:00:00',
        '2024-01-04T10:00:00'
    ], dtype='datetime64[ns]')
}

# Save as npz
np.savez(data_dir / "user.npz", **user_data)
np.savez(data_dir / "transaction.npz", **transaction_data)

# Create metadata.yaml
metadata = {
    'dataset_name': 'example4_cutoff_time',
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
                {'name': 'timestamp', 'dtype': 'datetime'}
            ],
            'time_column': 'timestamp'
        }
    ]
}

with open(example_dir / "metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Create target table with cutoff times
target_df = pd.DataFrame({
    'user_id': [1, 1, 2],
    'cutoff_time': pd.to_datetime([
        '2024-01-02 10:00:00',  # Before transaction 2 and 3
        '2024-01-04 00:00:00',   # Before transaction 3
        '2024-01-05 00:00:00'    # After all transactions
    ]),
    'target_label': [0, 0, 1]
})

target_df.to_csv(example_dir / "target_table.csv", index=False)

print(f"Created Example 4 in: {example_dir}")
print(f"\nUser table: {len(user_data['user_id'])} users")
print(f"Transaction table: {len(transaction_data['transaction_id'])} transactions")

# Print transaction breakdown
print(f"\nTransaction breakdown by user:")
for user_id in [1, 2]:
    mask = transaction_data['user_id'] == user_id
    amounts = transaction_data['amount'][mask]
    timestamps = transaction_data['timestamp'][mask]
    print(f"  User {user_id}: {len(amounts)} transactions")
    for i, (amt, ts) in enumerate(zip(amounts, timestamps)):
        print(f"    Transaction {transaction_data['transaction_id'][mask][i]}: amount={amt}, timestamp={pd.Timestamp(ts)}")

print(f"\nTarget table with cutoff times:")
print(target_df)
print(f"\nExpected features for each row:")
print(f"\nRow 1 (User 1, cutoff=2024-01-02 10:00:00):")
print(f"  Only transaction 1 (2024-01-01 10:00:00) is before cutoff")
print(f"  Expected COUNT: 1")
print(f"  Expected MEAN: 10.0")
print(f"\nRow 2 (User 1, cutoff=2024-01-04 00:00:00):")
print(f"  Transactions 1 and 2 (before cutoff)")
print(f"  Expected COUNT: 2")
print(f"  Expected MEAN: 15.0 (mean of [10, 20])")
print(f"\nRow 3 (User 2, cutoff=2024-01-05 00:00:00):")
print(f"  All transactions (4 and 5) are before cutoff")
print(f"  Expected COUNT: 2")
print(f"  Expected MEAN: 20.0 (mean of [15, 25])")

golden_data = {
    "user_id": np.array([1, 1, 2], dtype=np.int64),
    "cutoff_time": np.array([
        "2024-01-02T10:00:00.000000000",
        "2024-01-04T00:00:00.000000000",
        "2024-01-05T00:00:00.000000000",
    ], dtype="datetime64[ns]"),
    "target_label": np.array([0, 0, 1], dtype=np.int64),
    "user.age": np.array([25.0, 25.0, 30.0], dtype=np.float32),
    "user.COUNT(transaction)": np.array([1, 3, 2], dtype=np.int64),
    "user.MEAN(transaction.amount)": np.array([10.0, 20.0, 20.0], dtype=np.float32),
}

data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)
np.savez_compressed(data_dir / "golden_depth_2.npz", **golden_data)
print(f"Saved golden_depth_2.npz to {data_dir}")