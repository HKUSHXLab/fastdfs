from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import pydantic
import logging
from collections import defaultdict
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class CanonicalizeNumericConfig(pydantic.BaseModel):
    pass

@column_transform
class CanonicalizeNumericTransform(ColumnTransform):
    """Cast column data type to its canonical type."""
    config_class = CanonicalizeNumericConfig
    name = "canonicalize_numeric"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        pass

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        column.data = column.data.astype('float32')
        return [column]

class CanonicalizeDatetimeConfig(pydantic.BaseModel):
    pass

@column_transform
class CanonicalizeDatetimeTransform(ColumnTransform):
    """Cast column data type to its canonical type."""
    config_class = CanonicalizeDatetimeConfig
    name = "canonicalize_datetime"
    input_dtype = DBBColumnDType.datetime_t
    output_dtypes = [DBBColumnDType.datetime_t]
    output_name_formatters : List[str] = ["{name}"]

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        pass

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        column.data = column.data.astype('datetime64[ns]')
        return [column]
