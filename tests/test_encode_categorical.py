from loguru import logger

logger.enable("fastdfs")

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from fastdfs.transform.encode_categorical import (
    EncodeCategoryColumns,
    encode_series_with_label_encoder,
)
from fastdfs.transform.type_transform import CanonicalizeTypes
from fastdfs.dataset.meta import RDBTableSchema, RDBColumnSchema, RDBColumnDType, RDBTableDataFormat


@pytest.fixture
def category_metadata():
    return RDBTableSchema(
        name="items",
        source="dummy",
        format=RDBTableDataFormat.PARQUET,
        columns=[
            RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
            RDBColumnSchema(name="color", dtype=RDBColumnDType.category_t),
            RDBColumnSchema(name="score", dtype=RDBColumnDType.float_t),
        ],
    )


def test_encode_category_columns_basic(category_metadata):
    df = pd.DataFrame({
        "id": ["1", "2", "3"],
        "color": ["red", "blue", "red"],
        "score": [1.0, 2.0, 3.0],
    })
    transform = EncodeCategoryColumns()
    new_df, new_meta = transform(df, category_metadata)

    assert pd.api.types.is_float_dtype(new_df["color"])
    assert new_meta.column_dict["color"].dtype == RDBColumnDType.float_t
    assert new_meta.column_dict["score"].dtype == RDBColumnDType.float_t
    assert ("items", "color") in transform.encoders


def test_encode_series_unseen_label():
    le = LabelEncoder()
    le.fit(["a", "b"])
    out = encode_series_with_label_encoder(pd.Series(["a", "c", "b"]), le)
    assert out.iloc[0] == 0.0
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 1.0


def test_reuse_encoders_no_refit(category_metadata):
    df_train = pd.DataFrame({
        "id": ["1", "2"],
        "color": ["red", "blue"],
        "score": [1.0, 2.0],
    })
    transform = EncodeCategoryColumns()
    transform(df_train, category_metadata)
    encoders = dict(transform.encoders)

    transform2 = EncodeCategoryColumns(encoders=encoders)
    df_test = pd.DataFrame({
        "id": ["3"],
        "color": ["green"],
        "score": [3.0],
    })
    new_df, _ = transform2(df_test, category_metadata)
    assert pd.isna(new_df["color"].iloc[0])
    assert len(transform2.encoders[("items", "color")].classes_) == 2


def test_no_category_columns_unchanged(category_metadata):
    meta = RDBTableSchema(
        name="nums",
        source="dummy",
        format=RDBTableDataFormat.PARQUET,
        columns=[
            RDBColumnSchema(name="id", dtype=RDBColumnDType.primary_key),
            RDBColumnSchema(name="score", dtype=RDBColumnDType.float_t),
        ],
    )
    df = pd.DataFrame({"id": ["1"], "score": [1.0]})
    transform = EncodeCategoryColumns()
    new_df, new_meta = transform(df, meta)
    assert new_df["score"].iloc[0] == 1.0
    assert new_meta.column_dict["score"].dtype == RDBColumnDType.float_t


def test_encode_then_canonicalize_leaves_float(category_metadata):
    df = pd.DataFrame({
        "id": ["1", "2"],
        "color": ["red", "blue"],
        "score": [1.0, 2.0],
    })
    encoded_df, encoded_meta = EncodeCategoryColumns()(df, category_metadata)
    final_df, _ = CanonicalizeTypes()(encoded_df, encoded_meta)
    assert pd.api.types.is_float_dtype(final_df["color"])
    assert not pd.api.types.is_string_dtype(final_df["color"])
    assert final_df["color"].notna().all()
