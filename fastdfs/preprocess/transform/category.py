import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
from loguru import logger
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)

class RemapCategoryTransformConfig(pydantic.BaseModel):
    pass

@column_transform
class RemapCategoryTransform(ColumnTransform):
    config_class = RemapCategoryTransformConfig
    name = "remap_category"
    input_dtype = DBBColumnDType.category_t
    output_dtypes = [DBBColumnDType.category_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : RemapCategoryTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")
        _, self.categories = pd.factorize(column.data, use_na_sentinel=True)
        self.unseen_category = len(self.categories)

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        if column.data.ndim > 1:
            raise ValueError("RemapCategoryTransform only supports 1D data.")

        new_data = pd.Categorical(column.data, categories=self.categories).codes.copy()
        new_data[new_data == -1] = self.unseen_category
        new_data = new_data.astype('int64')
        new_meta = copy.deepcopy(column.metadata)
        new_meta['num_categories'] = len(self.categories) + 1

        return [ColumnData(new_meta, new_data)]
