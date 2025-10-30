# Example 4: Cutoff Time Handling

## Purpose
This example tests temporal filtering behavior - ensuring DFS only uses data before the cutoff time to prevent data leakage.

## Schema
- **user** table: Contains user attributes
  - `user_id` (primary_key)
  - `age` (float)

- **transaction** table: Time-stamped transactions
  - `transaction_id` (primary_key)
  - `user_id` (foreign_key -> user.user_id)
  - `amount` (float)
  - `timestamp` (datetime) - time column

- **target** table: What we want to predict on (with cutoff times)
  - `user_id` (links to user.user_id)
  - `cutoff_time` (datetime) - when prediction is made
  - `target_label` (what we want to predict)

## Test Data Setup
Transactions with timestamps:
- User 1: 
  - Transaction 1: amount=10.0, timestamp=2024-01-01 10:00:00
  - Transaction 2: amount=20.0, timestamp=2024-01-02 11:00:00
  - Transaction 3: amount=30.0, timestamp=2024-01-03 12:00:00

- User 2:
  - Transaction 4: amount=15.0, timestamp=2024-01-01 14:00:00
  - Transaction 5: amount=25.0, timestamp=2024-01-04 10:00:00

Target table with different cutoff times:
- Row 1: user_id=1, cutoff_time=2024-01-02 10:00:00 (before transaction 2 and 3)
- Row 2: user_id=1, cutoff_time=2024-01-04 00:00:00 (before transaction 3)
- Row 3: user_id=2, cutoff_time=2024-01-05 00:00:00 (after all transactions)

## Expected DFS Features (max_depth=1, agg_primitives=["count", "mean"], use_cutoff_time=True)

### Row 1: User 1, cutoff=2024-01-02 10:00:00
Only transaction 1 (timestamp=2024-01-01 10:00:00) is before cutoff:
- `user.COUNT(transaction)`: 1 (only transaction 1)
- `user.MEAN(transaction.amount)`: 10.0

### Row 2: User 1, cutoff=2024-01-04 00:00:00
Transactions 1 and 2 are before cutoff:
- `user.COUNT(transaction)`: 2 (transactions 1 and 2)
- `user.MEAN(transaction.amount)`: 15.0 (mean of [10, 20])

### Row 3: User 2, cutoff=2024-01-05 00:00:00
All transactions (4 and 5) are before cutoff:
- `user.COUNT(transaction)`: 2
- `user.MEAN(transaction.amount)`: 20.0 (mean of [15, 25])

## Key Testing Points
1. Cutoff time correctly filters transactions (strict less-than comparison)
2. Each target row gets features computed only from data before its cutoff time
3. Features correctly reflect temporal constraints
4. Boundary conditions (exact timestamp matches) are handled correctly
