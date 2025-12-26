import pytest
import sqlite3
import pandas as pd
from fastdfs.adapter.sqlite import SQLiteAdapter

def test_sqlite_adapter_hints(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    
    # Create tables without explicit PKs in schema to test overrides
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    conn.execute("CREATE TABLE orders (order_id INTEGER, user_id INTEGER, amount REAL, created_at TEXT)")
    
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO orders VALUES (101, 1, 50.0, '2023-01-01 10:00:00')")
    conn.commit()
    conn.close()

    # Define hints
    primary_keys = {"users": "id", "orders": "order_id"}
    foreign_keys = [("orders", "user_id", "users", "id")]
    time_columns = {"orders": "created_at"}
    type_hints = {"users": {"name": "category"}}

    adapter = SQLiteAdapter(
        db_path,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        time_columns=time_columns,
        type_hints=type_hints
    )
    
    rdb = adapter.load()
    
    # Verify PKs
    assert rdb.metadata.get_table_schema("users").primary_key == "id"
    assert rdb.metadata.get_table_schema("orders").primary_key == "order_id"
    
    # Verify FKs
    orders_meta = rdb.metadata.get_table_schema("orders")
    fk_found = False
    for col in orders_meta.columns:
        if col.name == "user_id":
            assert col.link_to == "users.id"
            fk_found = True
    assert fk_found
    
    # Verify Time Column
    assert orders_meta.time_column == "created_at"
    
    # Verify Type Hint
    users_meta = rdb.metadata.get_table_schema("users")
    for col in users_meta.columns:
        if col.name == "name":
            assert col.dtype == "category"