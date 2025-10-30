# Example 3: Multi-Level Aggregation (Depth 2)

## Purpose
This example tests aggregation features at depth 2 - aggregating through intermediate tables to reach grandchildren tables.

## Schema
- **user** table: Parent entity
  - `user_id` (primary_key)
  - `age` (float)

- **order** table: Child of user, parent of order_item
  - `order_id` (primary_key)
  - `user_id` (foreign_key -> user.user_id)
  - `order_date` (datetime)

- **order_item** table: Child of order, grandchild of user
  - `order_item_id` (primary_key)
  - `order_id` (foreign_key -> order.order_id)
  - `price` (float)
  - `quantity` (int)

- **target** table: What we want to predict on
  - `user_id` (links to user.user_id)
  - `target_label` (what we want to predict)

## Test Data Setup
- User 1:
  - Order 1: 2 items (price=[10.0, 20.0], quantity=[2, 1])
  - Order 2: 1 item (price=[15.0], quantity=[3])
  - Total items: 3 across 2 orders

- User 2:
  - Order 3: 1 item (price=[30.0], quantity=[1])
  - Total items: 1 across 1 order

## Expected DFS Features (max_depth=2, agg_primitives=["count", "mean", "max", "min"])

### Depth 1 Features (direct children):
- `user.COUNT(order)` - Number of orders per user
  - User 1: 2
  - User 2: 1

### Depth 2 Features (grandchildren):
- `user.COUNT(order_item)` - Total items across all orders
  - User 1: 3 (2 from order 1 + 1 from order 2)
  - User 2: 1

- `user.MEAN(order_item.price)` - Average item price across all orders
  - User 1: 15.0 (mean of [10, 20, 15])
  - User 2: 30.0

- `user.MEAN(order_item.quantity)` - Average quantity across all items
  - User 1: 2.0 (mean of [2, 1, 3])
  - User 2: 1.0

- `user.MAX(order_item.price)` - Maximum item price
  - User 1: 20.0
  - User 2: 30.0

- `user.MIN(order_item.price)` - Minimum item price
  - User 1: 10.0
  - User 2: 30.0

## Key Testing Points
1. Aggregation correctly traverses through intermediate tables (depth 2)
2. Features correctly aggregate grandchild entities
3. Intermediate aggregations (order level) also work correctly
4. Multiple aggregation paths (order -> order_item) are handled properly
