# DFS Minimal Examples

This directory contains minimal, focused examples to test individual important features of Deep Feature Synthesis (DFS).

## Purpose

These examples are designed to help you:
1. **Understand DFS step-by-step**: Each example focuses on a single DFS concept
2. **Verify correctness**: Compare fastdfs results with your manual calculations
3. **Debug issues**: Isolate specific DFS features when troubleshooting

## Examples

### Example 1: Direct Features (Depth 0)
**Location**: `example1_direct_features/`

Tests the most basic DFS functionality - retrieving direct attributes from related tables without any aggregation.

**Key Concepts**: Direct feature extraction, key mappings

---

### Example 2: Single-Level Aggregation (Depth 1)
**Location**: `example2_single_aggregation/`

Tests aggregation features at depth 1 - aggregating directly related child tables.

**Key Concepts**: COUNT, MEAN, MAX, MIN, MODE primitives, parent-child aggregation

---

### Example 3: Multi-Level Aggregation (Depth 2)
**Location**: `example3_multi_level_aggregation/`

Tests aggregation features at depth 2 - aggregating through intermediate tables to reach grandchildren tables.

**Key Concepts**: Multi-hop relationships, depth traversal, intermediate tables

---

### Example 4: Cutoff Time Handling
**Location**: `example4_cutoff_time/`

Tests temporal filtering behavior - ensuring DFS only uses data before the cutoff time to prevent data leakage.

**Key Concepts**: Temporal filtering, cutoff time, data leakage prevention

---

### Example 5: Multiple Relationships
**Location**: `example5_multiple_relationships/`

Tests DFS when the target table has multiple relationships (both user and item), ensuring features from both paths are generated correctly.

**Key Concepts**: Multiple key mappings, simultaneous feature generation from multiple paths

---

## Running the Examples

### Quick Start

Run all examples at once:
```bash
cd /root/linjie/fastdfs/examples/dfs_minimal_examples
conda activate tabular-chat-predictor
python run_all_examples.py
```

### Running Individual Examples

Each example directory contains:
- `metadata.yaml` - RDB schema definition
- `data/` - NPZ files with table data
- `target_table.csv` - Target table for prediction
- `README.md` - Detailed explanation and expected results
- `create_example.py` - Script to regenerate the example data

To run a single example programmatically:
```python
import fastdfs
import pandas as pd
from pathlib import Path

# Load example
example_dir = Path("example1_direct_features")
rdb = fastdfs.load_rdb(str(example_dir))
target_df = pd.read_csv(example_dir / "target_table.csv")

# Generate features
features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id"},
    config_overrides={
        "max_depth": 0,
        "agg_primitives": []
    }
)
```

## Understanding the Output

For each example, you'll see:
1. **RDB Information**: Tables loaded and their relationships
2. **Target Table**: The input data you want to predict on
3. **DFS Configuration**: Settings used (max_depth, primitives, etc.)
4. **Generated Features**: List of all new features created
5. **Full Result**: Complete dataframe with original + new features

## Comparing with Expected Results

Each example's `README.md` contains:
- Detailed test data setup
- **Expected DFS Features**: What features should be generated
- **Expected Values**: Specific values for sample rows
- **Key Testing Points**: What aspects of DFS are being tested

Compare the fastdfs output with these expected results to verify correctness.

## Next Steps

After running these examples:

1. **Manual Verification**: Calculate the expected features manually for each example
2. **Feature Analysis**: Review each generated feature to understand how DFS creates it
3. **Parameter Exploration**: Modify `max_depth` and `agg_primitives` to see how features change
4. **Compare Implementations**: Use these examples to compare fastdfs with other DFS implementations

## Example Structure

```
example1_direct_features/
├── README.md              # Explanation and expected results
├── metadata.yaml          # RDB schema
├── data/
│   └── user.npz          # User table data
├── target_table.csv       # Target table
└── create_example.py      # Script to regenerate data

example2_single_aggregation/
├── README.md
├── metadata.yaml
├── data/
│   ├── user.npz
│   └── transaction.npz
├── target_table.csv
└── create_example.py

...
```

## Tips for Learning DFS

1. **Start with Example 1**: Understand direct features before moving to aggregations
2. **Increment Depth Gradually**: Example 1 → Example 2 → Example 3
3. **Check Each Primitive**: Example 2 tests different primitives - verify each one
4. **Understand Temporal Logic**: Example 4 is crucial for preventing data leakage
5. **Multiple Relationships**: Example 5 shows how DFS handles complex scenarios

## Troubleshooting

If features don't match expected results:
1. Check the `key_mappings` - they must correctly link target to RDB
2. Verify `cutoff_time_column` is set correctly for temporal examples
3. Ensure `max_depth` is sufficient for the relationships you want
4. Check that `agg_primitives` includes the primitives you need
5. Review the RDB relationships in `metadata.yaml`

---

**Happy Learning!** 🚀
