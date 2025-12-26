
import pytest
import pandas as pd
import numpy as np
from fastdfs.transform.type_transform import CanonicalizeTypes
from fastdfs.dataset.meta import RDBTableSchema, RDBColumnSchema, RDBColumnDType, RDBTableDataFormat

@pytest.fixture
def sample_metadata():
    return RDBTableSchema(
        name="test_table",
        source="dummy",
        format=RDBTableDataFormat.PARQUET,
        columns=[
            RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
            RDBColumnSchema(name="value_float", dtype=RDBColumnDType.float_t),
            RDBColumnSchema(name="value_int", dtype=RDBColumnDType.timestamp_t),
            RDBColumnSchema(name="date", dtype=RDBColumnDType.datetime_t),
            RDBColumnSchema(name="category", dtype=RDBColumnDType.category_t),
            RDBColumnSchema(name="text", dtype=RDBColumnDType.text_t),
        ]
    )

def test_canonicalize_types_basic(sample_metadata):
    """Test basic type conversion."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value_float": ["1.1", "2.2", "3.3"],
        "value_int": ["10", "20", "30"],
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "category": ["A", "B", "A"],
        "text": [100, 200, 300],
        "extra_col": ["x", "y", "z"]  # Should be dropped
    })

    transform = CanonicalizeTypes()
    new_df, new_meta = transform(df, sample_metadata)

    # Check columns
    assert "extra_col" not in new_df.columns
    assert list(new_df.columns) == ["id", "value_float", "value_int", "date", "category", "text"]

    # Check types
    assert pd.api.types.is_string_dtype(new_df["id"])
    assert abs(new_df["value_float"].iloc[0] - 1.1) < 1e-6
    assert new_df["value_int"].iloc[0] == 10
    assert new_df["text"].iloc[0] == "100"

def test_canonicalize_types_missing_column(sample_metadata):
    """Test error raised when metadata column is missing."""
    df = pd.DataFrame({
        "id": [1, 2],
        # Missing other columns
    })

    transform = CanonicalizeTypes()
    with pytest.raises(ValueError, match="missing from the actual data"):
        transform(df, sample_metadata)

def test_canonicalize_types_coercion(sample_metadata):
    """Test that invalid values are coerced to NaN/NaT."""
    df = pd.DataFrame({
        "id": [1, 2],
        "value_float": ["1.1", "not_a_number"],
        "value_int": ["10", "bad_int"],
        "date": ["2023-01-01", "not_a_date"],
        "category": ["A", "B"],
        "text": ["ok", "ok"]
    })

    transform = CanonicalizeTypes()
    new_df, _ = transform(df, sample_metadata)

    assert pd.isna(new_df["value_float"].iloc[1])
    assert pd.isna(new_df["value_int"].iloc[1])
    assert pd.isna(new_df["date"].iloc[1])

def test_canonicalize_types_nullable_int(sample_metadata):
    """Test that integer columns handle NaNs (using Int64)."""
    df = pd.DataFrame({
        "id": [1, 2],
        "value_float": [1.1, 2.2],
        "value_int": [10, None],  # Contains None
        "date": ["2023-01-01", "2023-01-02"],
        "category": ["A", "B"],
        "text": ["a", "b"]
    })

    transform = CanonicalizeTypes()
    new_df, _ = transform(df, sample_metadata)

    assert pd.isna(new_df["value_int"].iloc[1])
    assert new_df["value_int"].dtype == "Int64"  # Nullable Int64
