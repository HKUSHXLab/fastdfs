from typing import Tuple, Dict, Optional, List, Union
from pathlib import Path
import featuretools as ft
import os
import pandas as pd

import duckdb
from ..dataset.meta import RDBCutoffTimeColumn

class DuckDBBuilder:
    def __init__(self, path : Path):
        self.path = path
        if os.path.exists(self.path):
            os.remove(self.path)
        self.db = duckdb.connect(str(self.path))
        self.time_columns = {}
        self.cutoff_time = None


    def add_dataframe(
        self,
        dataframe_name : str,
        dataframe : pd.DataFrame,
        index : str,
        time_index : Optional[str] = None,
        logical_types : Optional[Dict[str, str]] = None,
        semantic_tags : Optional[Dict[str, str]] = None,
    ):
        self.db.sql(f"CREATE TABLE \"{dataframe_name}\" AS SELECT * from dataframe")
        if time_index is not None:
            self.time_columns[dataframe_name] = time_index

    def add_relationship(
        self,
        pk_table : str,
        pk_column : str,
        fk_table : str,
        fk_column : str,
    ):
        pass

    def set_cutoff_time(
        self,
        cutoff_time : Optional[pd.DataFrame]
    ):
        if cutoff_time is None:
            return
        assert RDBCutoffTimeColumn.column_name in cutoff_time.columns
        self.cutoff_time_col_name = RDBCutoffTimeColumn.column_name
        self.cutoff_time_table_name = RDBCutoffTimeColumn.table_name
        self.db.sql(f"CREATE TABLE {self.cutoff_time_table_name} AS SELECT * from cutoff_time")
        self.cutoff_time = cutoff_time


    def set_task_index(
        self,
        name: str,
        index : pd.Series
    ):
        self.index_name = name
        self.index = index
