from typing import Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
import pydantic
import logging
from collections import defaultdict
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo
from ...utils import datetime_utils
from .base import (
    RDBTransform,
    rdb_transform,
    ColumnData,
    RDBData,
    is_task_table,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FilterColumnConfig(pydantic.BaseModel):
    drop_dtypes: Optional[List[str]] = None
    drop_redundant: bool = True

@rdb_transform
class FilterColumn(RDBTransform):
    """Filter columns.

    Criterion:
    - Non-key columns with identical values.
    """

    config_class = FilterColumnConfig
    name = "filter_column"

    def __init__(self, config : FilterColumnConfig):
        super().__init__(config)

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        self.columns_to_filer = defaultdict(list)
        for tbl_name, tbl in rdb_data.tables.items():
            for col_name, col in tbl.items():
                if col.metadata['dtype'] in [
                    DBBColumnDType.primary_key,
                    DBBColumnDType.foreign_key,
                ]:
                    continue
                if col.data.ndim > 1:
                    # Ignore vector embeddings.
                    continue
                logger.info(f"Fitting filter_column for {col_name}.")

                if self.config.drop_redundant and col.metadata['dtype'] not in [
                    DBBColumnDType.text_t
                ]:
                    unique_data = pd.unique(col.data)
                    if len(unique_data) <= 1:
                        self.columns_to_filer[tbl_name].append(col_name)
                
                if self.config.drop_dtypes is not None:
                    if col.metadata['dtype'] in self.config.drop_dtypes:
                        self.columns_to_filer[tbl_name].append(col_name)

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for tbl_name, tbl in rdb_data.tables.items():
            for col_name in self.columns_to_filer[tbl_name]:
                if col_name in tbl:
                    logger.info(f"Drop column {tbl_name}/{col_name}.")
                    tbl.pop(col_name)
        return rdb_data
