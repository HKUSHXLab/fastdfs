
from loguru import logger
logger.enable("fastdfs")
import pytest
import pandas as pd
import numpy as np
from fastdfs.api import create_rdb
from fastdfs.dataset.meta import RDBColumnDType, RDBMeta, RDBTableSchema, RDBTableDataFormat
from fastdfs.transform.infer_schema import InferSchemaTransform
from fastdfs.dataset.rdb import RDB

@pytest.fixture
def sample_data():
    users = pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 30, 35],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    
    items = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "price": [10.5, 20.0, 15.75],
        "category": ["cat1", "cat2", "cat1"]
    })
    
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "item_id": ["a", "b", "a", "c"],
        "timestamp": pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08"]),
        "rating": [5, 4, 3, 5]
    })
    
    return {
        "users": users,
        "items": items,
        "interactions": interactions
    }

def test_create_rdb_basic(sample_data):
    rdb = create_rdb(
        tables=sample_data,
        name="test_rdb",
        primary_keys={"users": "user_id", "items": "item_id"},
        foreign_keys=[
            ("interactions", "user_id", "users", "user_id"),
            ("interactions", "item_id", "items", "item_id")
        ],
        time_columns={"interactions": "timestamp"}
    )
    
    assert rdb.metadata.name == "test_rdb"
    assert set(rdb.table_names) == {"users", "items", "interactions"}
    
    # Check Users Schema
    users_meta = rdb.get_table_metadata("users")
    assert users_meta.column_dict["user_id"].dtype == RDBColumnDType.primary_key
    assert users_meta.column_dict["age"].dtype == RDBColumnDType.float_t
    assert users_meta.column_dict["signup_date"].dtype == RDBColumnDType.datetime_t
    
    # Check Interactions Schema
    interactions_meta = rdb.get_table_metadata("interactions")
    assert interactions_meta.column_dict["user_id"].dtype == RDBColumnDType.foreign_key
    assert interactions_meta.column_dict["user_id"].link_to == "users.user_id"
    assert interactions_meta.column_dict["timestamp"].dtype == RDBColumnDType.datetime_t
    assert interactions_meta.time_column == "timestamp"

def test_create_rdb_type_hints(sample_data):
    rdb = create_rdb(
        tables=sample_data,
        name="test_rdb_hints",
        type_hints={
            "users": {"age": "category"}
        }
    )
    
    users_meta = rdb.get_table_metadata("users")
    assert users_meta.column_dict["age"].dtype == RDBColumnDType.category_t


