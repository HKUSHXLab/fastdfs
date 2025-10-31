#!/usr/bin/env python3
"""
Create minimal example 5: Multiple Relationships
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
    'city': np.array(['NYC', 'SF'], dtype=object)
}

# Create item table
item_data = {
    'item_id': np.array([1, 2], dtype=object),
    'price': np.array([100.0, 50.0], dtype=np.float32),
    'category': np.array(['Electronics', 'Books'], dtype=object)
}

# Create interaction table (child of both user and item)
# User 1 + Item 1: 2 interactions
# User 1 + Item 2: 1 interaction
# User 2 + Item 2: 1 interaction
interaction_data = {
    'interaction_id': np.array([1, 2, 3, 4], dtype=object),
    'user_id': np.array([1, 1, 1, 2], dtype=object),
    'item_id': np.array([1, 1, 2, 2], dtype=object),
    'rating': np.array([4.5, 4.0, 3.5, 5.0], dtype=np.float32),
    'timestamp': np.array([
        '2024-01-01T10:00:00',
        '2024-01-02T11:00:00',
        '2024-01-03T12:00:00',
        '2024-01-04T13:00:00'
    ], dtype='datetime64[ns]')
}

# Save as npz
np.savez(data_dir / "user.npz", **user_data)
np.savez(data_dir / "item.npz", **item_data)
np.savez(data_dir / "interaction.npz", **interaction_data)

# Create metadata.yaml
metadata = {
    'dataset_name': 'example5_multiple_relationships',
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
        },
        {
            'name': 'item',
            'source': 'data/item.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'item_id', 'dtype': 'primary_key'},
                {'name': 'price', 'dtype': 'float'},
                {'name': 'category', 'dtype': 'category'}
            ]
        },
        {
            'name': 'interaction',
            'source': 'data/interaction.npz',
            'format': 'numpy',
            'columns': [
                {'name': 'interaction_id', 'dtype': 'primary_key'},
                {'name': 'user_id', 'dtype': 'foreign_key', 'link_to': 'user.user_id'},
                {'name': 'item_id', 'dtype': 'foreign_key', 'link_to': 'item.item_id'},
                {'name': 'rating', 'dtype': 'float'},
                {'name': 'timestamp', 'dtype': 'datetime'}
            ],
            'time_column': 'timestamp'
        }
    ]
}

with open(example_dir / "metadata.yaml", 'w') as f:
    yaml.dump(metadata, f, default_flow_style=False)

# Create target table with both user_id and item_id
target_df = pd.DataFrame({
    'user_id': [1, 1, 2],
    'item_id': [1, 1, 2],
    'cutoff_time': pd.to_datetime([
        '2024-01-02 10:00:00',  # Before interaction 2
        '2024-01-05 00:00:00',  # After all interactions
        '2024-01-05 00:00:00'   # After all interactions
    ]),
    'target_label': [0, 0, 1]
})

target_df.to_csv(example_dir / "target_table.csv", index=False)

print(f"Created Example 5 in: {example_dir}")
print(f"\nUser table: {len(user_data['user_id'])} users")
for i, user_id in enumerate(user_data['user_id']):
    print(f"  User {user_id}: age={user_data['age'][i]}, city={user_data['city'][i]}")

print(f"\nItem table: {len(item_data['item_id'])} items")
for i, item_id in enumerate(item_data['item_id']):
    print(f"  Item {item_id}: price={item_data['price'][i]}, category={item_data['category'][i]}")

print(f"\nInteraction table: {len(interaction_data['interaction_id'])} interactions")
for i in range(len(interaction_data['interaction_id'])):
    print(f"  Interaction {interaction_data['interaction_id'][i]}: "
          f"user_id={interaction_data['user_id'][i]}, "
          f"item_id={interaction_data['item_id'][i]}, "
          f"rating={interaction_data['rating'][i]}, "
          f"timestamp={pd.Timestamp(interaction_data['timestamp'][i])}")

print(f"\nTarget table:")
print(target_df)

print(f"\nExpected features by row:")
print(f"\nRow 1 (user_id=1, item_id=1, cutoff=2024-01-02 10:00:00):")
print(f"  User 1 direct: age=25.0, city='NYC'")
print(f"  User 1 aggregated (before cutoff): COUNT=1, MEAN(rating)=4.5")
print(f"  Item 1 direct: price=100.0, category='Electronics'")
print(f"  Item 1 aggregated (before cutoff): COUNT=1, MEAN(rating)=4.5")

print(f"\nRow 2 (user_id=1, item_id=1, cutoff=2024-01-05 00:00:00):")
print(f"  User 1 aggregated: COUNT=3, MEAN(rating)=4.0 (mean of [4.5, 4.0, 3.5])")
print(f"  Item 1 aggregated: COUNT=2, MEAN(rating)=4.25 (mean of [4.5, 4.0])")

print(f"\nRow 3 (user_id=2, item_id=2, cutoff=2024-01-05 00:00:00):")
print(f"  User 2 direct: age=30.0, city='SF'")
print(f"  User 2 aggregated: COUNT=1, MEAN(rating)=5.0")
print(f"  Item 2 direct: price=50.0, category='Books'")
print(f"  Item 2 aggregated: COUNT=2, MEAN(rating)=4.25 (mean of [3.5, 5.0])")

golden_data = {
    "user_id": np.array([1, 1, 2], dtype=np.int64),
    "item_id": np.array([1, 1, 2], dtype=np.int64),
    "cutoff_time": np.array([
        "2024-01-02T10:00:00.000000000",
        "2024-01-05T00:00:00.000000000",
        "2024-01-05T00:00:00.000000000",
    ], dtype="datetime64[ns]"),
    "target_label": np.array([0, 0, 1], dtype=np.int64),
    "user.age": np.array([25.0, 25.0, 30.0], dtype=np.float32),
    "user.city": np.array(['NYC', 'NYC', 'SF'], dtype=object),
    "item.price": np.array([100.0, 100.0, 50.0], dtype=np.float32),
    "item.category": np.array(['Electronics', 'Electronics', 'Books'], dtype=object),
    "user.COUNT(interaction)": np.array([1, 3, 1], dtype=np.int64),
    "user.MEAN(interaction.rating)": np.array([4.5, 4.0, 5.0], dtype=np.float32),
    "user.STD(interaction.rating)": np.array([np.nan, 0.5, np.nan], dtype=np.float32),
    "item.COUNT(interaction)": np.array([1, 2, 2], dtype=np.int64),
    "item.MEAN(interaction.rating)": np.array([4.5, 4.25, 4.25], dtype=np.float32),
    "item.STD(interaction.rating)": np.array([np.nan, 0.353553, 1.060660], dtype=np.float32),
}

data_dir = example_dir / "data"
data_dir.mkdir(exist_ok=True)
np.savez_compressed(data_dir / "golden_depth_2.npz", **golden_data)
print(f"Saved golden_depth_2.npz to {data_dir}")