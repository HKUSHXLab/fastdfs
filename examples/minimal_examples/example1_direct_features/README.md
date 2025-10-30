# Example 1: Direct Features (Depth 1)

## Purpose
This example tests the most basic DFS functionality: retrieving direct attributes from related tables without any aggregation.

## Important Note on DFS Depth Levels
- **max_depth=0**: Only identity features (columns of the target entity itself). **No features from related tables.**
- **max_depth=1**: Identity features + direct features from parent entities (like `user.age`) + aggregations on direct children

To retrieve attributes from related tables like `user.age`, we need **max_depth=1**, not 0.

## Schema
- **user** table: Contains user attributes
  - `user_id` (primary_key)
  - `age` (float)
  - `city` (category)

- **target** table: What we want to predict on
  - `user_id` (links to user.user_id)
  - `target_label` (what we want to predict)

## Expected DFS Features (max_depth=1, no aggregation primitives)
With `max_depth=1` and `agg_primitives=[]`, DFS will retrieve direct features from the user table:
- `user.age` - Direct attribute from user table
- `user.city` - Direct attribute from user table

## Key Testing Points
1. Direct feature extraction works correctly at depth 1
2. Key mappings properly link target to user table
3. No aggregation features are generated when agg_primitives is empty
