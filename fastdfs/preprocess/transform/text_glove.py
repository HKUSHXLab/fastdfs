from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
from loguru import logger
from ...dataset import DBBColumnDType

from ...utils.device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from tqdm import tqdm
import torch
import torch.multiprocessing as mp

import gensim.downloader as api
from gensim.utils import tokenize

class GloveTextEmbeddingTransformConfig(pydantic.BaseModel):
    model_name : str = "glove-twitter"
    dim : int = 100
    max_num_procs : int = 16

def _run_one_proc(proc_id, model, dim, data):
    embeddings = []
    if proc_id == 0:
        iterator = tqdm(data)
    else:
        iterator = data
    for text in iterator:
        if not isinstance(text, str):
            embed = np.zeros(dim)
        else:
            token_list = list(tokenize(text))
            if len(token_list) == 0:
                embed = np.zeros(dim)
            else:
                embed = model.get_mean_vector(token_list)
        embeddings.append(embed)
    return np.stack(embeddings).astype('float32')

@column_transform
class GloveTextEmbeddingTransform(ColumnTransform):
    config_class = GloveTextEmbeddingTransformConfig
    name = "glove_text_embedding"
    input_dtype = DBBColumnDType.text_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : GloveTextEmbeddingTransformConfig):
        super().__init__(config)
        self.model = api.load(f"{config.model_name}-{config.dim}")
        assert self.model.vector_size == config.dim, \
            "Dimension of the model does not match the config."

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : self.config.dim,
        }

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        data = column.data
        num_procs = min(device.cpu_count // 2, self.config.max_num_procs)
        if num_procs > 1:
            logger.info("Spawn workers to generate embeddings using multi-CPU kernels.")
            ctx = mp.get_context('spawn')
            worklist = np.array_split(data, num_procs)
            with ctx.Pool(processes=num_procs) as pool:
                results = []
                for proc_id in range(num_procs):
                    rst = pool.apply_async(
                        _run_one_proc,
                        (proc_id, self.model, self.config.dim, worklist[proc_id])
                    )
                    results.append(rst)
                results = [rst.get() for rst in results]
            new_data = np.concatenate(results, axis=0)
        else:
            new_data = _run_one_proc(self.model, self.config.dim, data)
        return [ColumnData(self.new_meta, new_data)]
