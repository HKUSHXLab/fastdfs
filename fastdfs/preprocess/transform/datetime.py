from enum import Enum
import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
from loguru import logger
import datetime
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from ...utils import datetime_utils

class DatetimeFeaturizeMethod(str, Enum):
    YEAR = 'YEAR'
    MONTH = 'MONTH'
    DAY = 'DAY'
    DAYOFWEEK = 'DAYOFWEEK'
    TIMESTAMP = 'TIMESTAMP'

_METHOD_TO_FUNC = {
    'YEAR' : datetime_utils.dt2year,
    'MONTH' : datetime_utils.dt2month,
    'DAY' : datetime_utils.dt2day,
    'DAYOFWEEK' : datetime_utils.dt2dayofweek,
    'TIMESTAMP' : lambda data : datetime_utils.dt2ts(data).astype('float32')
}

_METHOD_TO_OUTPUT_DTYPE = {
    'YEAR' : DBBColumnDType.category_t,
    'MONTH' : DBBColumnDType.category_t,
    'DAY' : DBBColumnDType.category_t,
    'DAYOFWEEK' : DBBColumnDType.category_t,
    'TIMESTAMP' : DBBColumnDType.float_t,
}

class FeaturizeDatetimeTransformConfig(pydantic.BaseModel):
    methods : List[DatetimeFeaturizeMethod]
    retain_original : bool = False

@column_transform
class FeaturizeDatetimeTransform(ColumnTransform):
    config_class = FeaturizeDatetimeTransformConfig
    name = "featurize_datetime"
    input_dtype = DBBColumnDType.datetime_t

    def __init__(self, config : FeaturizeDatetimeTransformConfig):
        super().__init__(config)

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
        new_cols = []
        self.output_dtypes = []
        self.output_name_formatters = []

        for method in self.config.methods:
            output_dtype = _METHOD_TO_OUTPUT_DTYPE[method]
            new_meta = {'dtype' : output_dtype}
            new_data = _METHOD_TO_FUNC[method](column.data)
            new_cols.append(ColumnData(new_meta, new_data))
            self.output_dtypes.append(output_dtype)
            self.output_name_formatters.append(method + '_{name}')

        if self.config.retain_original:
            new_cols.append(column)
            self.output_dtypes.append(DBBColumnDType.datetime_t)
            self.output_name_formatters.append('{name}')
        elif column.metadata.get('is_time_column', False):
            ts_meta = {'dtype' : DBBColumnDType.timestamp_t, 'is_time_column' : True}
            ts = datetime_utils.dt2ts(column.data)
            new_cols.append(ColumnData(ts_meta, ts))
            self.output_dtypes.append(DBBColumnDType.timestamp_t)
            self.output_name_formatters.append('{name}')

        return new_cols
