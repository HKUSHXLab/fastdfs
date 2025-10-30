# Example 5: Multiple Relationships

## Purpose
This example tests DFS when the target table has multiple relationships (both user and item relationships), ensuring features from both paths are generated correctly.

## Schema
- **user** table: Contains user attributes
  - `user_id` (primary_key)
  - `age` (float)
  - `city` (category)

- **item** table: Contains item attributes
  - `item_id` (primary_key)
  - `price` (float)
  - `category` (category)

- **interaction** table: Child of both user and item
  - `interaction_id` (primary_key)
  - `user_id` (foreign_key -> user.user_id)
  - `item_id` (foreign_key -> item.item_id)
  - `rating` (float)
  - `timestamp` (datetime)

- **target** table: What we want to predict on (has both user_id and item_id)
  - `user_id` (links to user.user_id)
  - `item_id` (links to item.item_id)
  - `cutoff_time` (datetime)
  - `target_label` (what we want to predict)

## Test Data Setup
Users:
- User 1: age=25, city='NYC'
- User 2: age=30, city='SF'

Items:
- Item 1: price=100.0, category='Electronics'
- Item 2: price=50.0, category='Books'

Interactions:
- User 1 + Item 1: rating=4.5, timestamp=2024-01-01 10:00:00
- User 1 + Item 1: rating=4.0, timestamp=2024-01-02 11:00:00
- User 1 + Item 2: rating=3.5, timestamp=2024-01-03 12:00:00
- User 2 + Item 2: rating=5.0, timestamp=2024-01-04 13:00:00

## Expected DFS Features (max_depth=1, agg_primitives=["count", "mean"], use_cutoff_time=True)

### Features from user relationship:
- `user.age` - Direct attribute
- `user.city` - Direct attribute
- `user.COUNT(interaction)` - Number of interactions per user (before cutoff)
- `user.MEAN(interaction.rating)` - Average rating given by user (before cutoff)

### Features from item relationship:
- `item.price` - Direct attribute
- `item.category` - Direct attribute
- `item.COUNT(interaction)` - Number of interactions per item (before cutoff)
- `item.MEAN(interaction.rating)` - Average rating received by item (before cutoff)

## Example Target Rows:
1. Row: user_id=1, item_id=1, cutoff_time=2024-01-02 10:00:00
   - User 1 features: COUNT=1, MEAN(rating)=4.5 (only first interaction before cutoff)
   - Item 1 features: COUNT=1, MEAN(rating)=4.5 (only first interaction before cutoff)

2. Row: user_id=1, item_id=1, cutoff_time=2024-01-05 00:00:00
   - User 1 features: COUNT=3, MEAN(rating)=4.0 (mean of [4.5, 4.0, 3.5])
   - Item 1 features: COUNT=2, MEAN(rating)=4.25 (mean of [4.5, 4.0])

3. Row: user_id=2, item_id=2, cutoff_time=2024-01-05 00:00:00
   - User 2 features: COUNT=1, MEAN(rating)=5.0
   - Item 2 features: COUNT=2, MEAN(rating)=4.25 (mean of [3.5, 5.0])

## Key Testing Points
1. Both user and item relationships generate features simultaneously
2. Features from different relationship paths don't interfere
3. Aggregations correctly filter by both user_id and item_id
4. Cutoff time works correctly for both relationship paths
5. Direct features from both parent tables are included
