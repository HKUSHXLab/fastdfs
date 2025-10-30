# Example 2: Single-Level Aggregation (Depth 1)

## Purpose
This example tests aggregation features at depth 1 - aggregating directly related child tables.

## Schema
- **user** table: Contains user attributes
  - `user_id` (primary_key)
  - `age` (float)

- **transaction** table: Child table linked to user
  - `transaction_id` (primary_key)
  - `user_id` (foreign_key -> user.user_id)
  - `amount` (float)
  - `category` (category)

- **target** table: What we want to predict on
  - `user_id` (links to user.user_id)
  - `target_label` (what we want to predict)

## Test Data Setup
- User 1: 3 transactions [10.0, 20.0, 30.0] in categories [A, A, B]
- User 2: 2 transactions [15.0, 25.0] in categories [A, B]
- User 3: 1 transaction [50.0] in category [B]

## Expected DFS Features (max_depth=1, agg_primitives=["count", "mean", "max", "min", "mode"])
For each user in target table:
- `user.COUNT(transaction)` - Number of transactions per user
  - User 1: 3
  - User 2: 2
  - User 3: 1

- `user.MEAN(transaction.amount)` - Average transaction amount
  - User 1: 20.0 (mean of [10, 20, 30])
  - User 2: 20.0 (mean of [15, 25])
  - User 3: 50.0

- `user.MAX(transaction.amount)` - Maximum transaction amount
  - User 1: 30.0
  - User 2: 25.0
  - User 3: 50.0

- `user.MIN(transaction.amount)` - Minimum transaction amount
  - User 1: 10.0
  - User 2: 15.0
  - User 3: 50.0

- `user.MODE(transaction.category)` - Most frequent category
  - User 1: A (appears twice)
  - User 2: A or B (tie, implementation dependent)
  - User 3: B

## Key Testing Points
1. COUNT primitive correctly counts related records
2. MEAN primitive correctly calculates averages
3. MAX/MIN primitives correctly find extremes
4. MODE primitive correctly finds most frequent values
5. Features correctly aggregate over parent-child relationships
