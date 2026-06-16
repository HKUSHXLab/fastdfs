"""
Label-encode RDB category columns to numeric codes for float-style DFS aggregations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from ..dataset.meta import RDBColumnDType, RDBColumnSchema, RDBTableSchema
from ..dataset.rdb import RDB
from .base import RDBTransform, RDBTransformWrapper, TableTransform


def encode_series_with_label_encoder(series: pd.Series, encoder: LabelEncoder) -> pd.Series:
    """Map values with ``encoder``; unseen labels become NaN."""
    as_str = series.astype(str)
    known = set(encoder.classes_)
    out = np.full(len(as_str), np.nan, dtype=np.float64)
    mask = as_str.isin(known)
    if mask.any():
        out[mask.to_numpy()] = encoder.transform(as_str[mask]).astype(np.float64)
    return pd.Series(out, index=series.index)


class EncodeCategoryColumns(TableTransform):
    """
    Replace ``category`` columns with label-encoded float codes and update schema to ``float``.

    Fit encoders on first use (per table/column). Reuse ``encoders`` at inference to avoid leakage
    from re-fitting on val/test RDB snapshots.
    """

    def __init__(self, encoders: Optional[Dict[Tuple[str, str], LabelEncoder]] = None):
        self.encoders: Dict[Tuple[str, str], LabelEncoder] = (
            dict(encoders) if encoders is not None else {}
        )

    def __call__(
        self, table: pd.DataFrame, table_metadata: RDBTableSchema
    ) -> Tuple[pd.DataFrame, RDBTableSchema]:
        new_table = table.copy()
        new_columns = []

        for col_schema in table_metadata.columns:
            col_name = col_schema.name
            if col_schema.dtype != RDBColumnDType.category_t or col_name not in new_table.columns:
                new_columns.append(col_schema)
                continue

            key = (table_metadata.name, col_name)
            if key not in self.encoders:
                le = LabelEncoder()
                le.fit(new_table[col_name].astype(str))
                self.encoders[key] = le
                logger.debug(
                    f"EncodeCategoryColumns: fitted {table_metadata.name}.{col_name} "
                    f"({len(le.classes_)} classes)"
                )

            new_table[col_name] = encode_series_with_label_encoder(
                new_table[col_name], self.encoders[key]
            )
            new_columns.append(
                RDBColumnSchema(name=col_name, dtype=RDBColumnDType.float_t)
            )

        new_metadata = RDBTableSchema(
            name=table_metadata.name,
            source=table_metadata.source,
            format=table_metadata.format,
            columns=new_columns,
            time_column=table_metadata.time_column,
        )
        return new_table, new_metadata


class EncodeCategoryColumnsRDB(RDBTransform):
    """Apply :class:`EncodeCategoryColumns` to every table in an RDB."""

    def __init__(self, encoders: Optional[Dict[Tuple[str, str], LabelEncoder]] = None):
        self._table_transform = EncodeCategoryColumns(encoders=encoders)
        self._wrapper = RDBTransformWrapper(self._table_transform)

    @property
    def encoders(self) -> Dict[Tuple[str, str], LabelEncoder]:
        return self._table_transform.encoders

    def __call__(self, rdb: RDB) -> RDB:
        return self._wrapper(rdb)
