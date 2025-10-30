#!/usr/bin/env python3
"""
Create minimal example 3: Multi-Level Aggregation (Depth 2)
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

# Create order table (child of user)
# User 1: 2 orders (order_id 1 and 2)
# User 2: 1 order (order_id 3)
order_data = {
    'order_id': np.array([1, 2, 3], dtype=object),
    'user_id': np.array([1, 1, 2], dtype=object),
    'order_date': np.array(['2024-01-01', '2024-01-02', '2024-01-03'], dtype='datetime64[D]')
}

# Create order_item table (child of order, grandchild of user)
# Order 1: 2 items (price=[10, 20], quantity=[2, 1])
# Order 2: 1 item (price=[15], quantity=[3])
# Order 3: 1 item (price=[30], quantity=[1])
order_item_data = {
    'order_item_id': np.array([1, 2, 3, 4], dtype=object),
    'order_id': np.array([1, 1, 2, 3], dtype=object),
    'price': np.array([10.0, 20.0, 15.0, 30.0], dtype=np.float32),
    'quantity': np.array([2, 1, 3, 1], dtype=np.int32)
}

# Save as npz
np.savez(data_dir / "user.npz", **user_data)
np.savez(data_dir / "order.npz", **order_data)
np.savez(data_dir / "order_item.npz", **order_item_data)

# Create metadata.yaml
metadata = {
    'dataset_name': 'example3_multi_level_aggregation',
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
            'name': 'order',
            'source': 'data/order.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'order_id', 'dtype': 'primary_key'},
                {'name': 'user_id', 'dtype': 'foreign_key', 'link_to': 'user.user_id'},
                {'name': 'order_date', 'dtype': 'datetime'}
            ],
            'time_column': 'order_date'
        },
        {
            'name': 'order_item',
            'source': 'data/order_item.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'order_item_id', 'dtype': 'primary_key'},
                {'name': 'order_id', 'dtype': 'foreign_key', 'link_to': 'order.order_id'},
                {'name': 'price', 'dtype': 'float'},
                {'name': 'quantity', 'dtype': 'float'}  # Using float for compatibility
            ]
        }
    ]
}

with open(example_dir / "metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Create target table
target_df = pd.DataFrame({
    'user_id': [1, 2, 1],
    'target_label': [0, 1, 0]
})

target_df.to_csv(example_dir / "target_table.csv", index=False)

print(f"Created Example 3 in: {example_dir}")
print(f"\nUser table: {len(user_data['user_id'])} users")

# Print breakdown
print(f"\nOrder breakdown by user:")
for user_id in [1, 2]:
    order_mask = order_data['user_id'] == user_id
    user_orders = order_data['order_id'][order_mask]
    print(f"  User {user_id}: {len(user_orders)} orders (order_ids: {user_orders.tolist()})")
    
    # For each order, find items
    total_items = 0
    all_prices = []
    all_quantities = []
    for order_id in user_orders:
        item_mask = order_item_data['order_id'] == order_id
        prices = order_item_data['price'][item_mask]
        quantities = order_item_data['quantity'][item_mask]
        total_items += len(prices)
        all_prices.extend(prices.tolist())
        all_quantities.extend(quantities.tolist())
        
    print(f"    Total items: {total_items}")
    print(f"    Prices: {all_prices}")
    print(f"    Quantities: {all_quantities}")
    print(f"    Expected COUNT(order): {len(user_orders)}")
    print(f"    Expected COUNT(order_item): {total_items}")
    print(f"    Expected MEAN(order_item.price): {np.mean(all_prices):.1f}")
    print(f"    Expected MEAN(order_item.quantity): {np.mean(all_quantities):.1f}")
    print(f"    Expected MAX(order_item.price): {np.max(all_prices):.1f}")
    print(f"    Expected MIN(order_item.price): {np.min(all_prices):.1f}")
    print()

print(f"Target table shape: {target_df.shape}")
print(f"Target table:\n{target_df}")


golden_data = {
   "user_id": np.array([1, 2, 1], dtype=np.int64),
   "target_label": np.array([0, 1, 0], dtype=np.int64),
   "user.age": np.array([25.0, 30.0, 25.0], dtype=float),
   "user.COUNT(order)": np.array([2, 1, 2], dtype=np.int64),
}

data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)
np.savez_compressed(data_dir / "golden_depth_2.npz", **golden_data)
print(f"Saved golden_depth_2.npz to {data_dir}")