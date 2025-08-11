from typing import Tuple, Dict, Optional, List
import numpy as np
import pydantic
import logging

from ...utils.device import DeviceInfo
from .base import (
    RDBTransform,
    rdb_transform,
    get_column_transform_class,
    ColumnData,
    RDBData,
)
from .wrapper import RDBTransformWrapper

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class _NameAndConfig(pydantic.BaseModel):
    name : str
    config : Optional[Dict] = {}

class ColumnTransformChainConfig(pydantic.BaseModel):
    transforms : List[_NameAndConfig]

@rdb_transform
class ColumnTransformChain(RDBTransform):

    config_class = ColumnTransformChainConfig
    name = "column_transform_chain"

    def __init__(self, config : ColumnTransformChainConfig):
        super().__init__(config)
        self.transforms = [
            RDBTransformWrapper(xcfg.name, xcfg.config)
            for xcfg in config.transforms
        ]
        logger.debug([xform.col_xform_config for xform in self.transforms])

    def fit(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ):
        self.fit_transform(rdb_data, device)

    def transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for xform in self.transforms:
            rdb_data = xform.transform(rdb_data, device)
        return rdb_data

    def fit_transform(
        self,
        rdb_data : RDBData,
        device : DeviceInfo
    ) -> RDBData:
        for xform in self.transforms:
            xform.fit(rdb_data, device)
            rdb_data = xform.transform(rdb_data, device)
        return rdb_data
