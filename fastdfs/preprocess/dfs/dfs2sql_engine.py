from typing import Optional, List
from pathlib import Path
import pandas as pd
import featuretools as ft
from sql_formatter.core import format_sql
from functools import reduce
import tqdm
import numpy as np
import pprint

from .core import DFSEngine, DFSConfig, dfs_engine
from .gen_sqls import features2sql, decode_column_from_sql
from .database import DuckDBBuilder
from loguru import logger


@dfs_engine
class DFS2SQLEngine(DFSEngine):
    name = "dfs2sql"

    def filter_nested_array_agg_features(
        self,
        features: List[ft.FeatureBase],
    ) -> List[ft.FeatureBase]:
        if len(features) == 0:
            return features
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        new_features = []
        for feat in features:
            feat_str = str(feat)
            agg_count = _check_array_agg_occurrences(feat_str, array_agg_func_names)
            if agg_count > 1:
                # Remove features with nested array aggregation
                continue
            new_features.append(feat)
        return new_features

    def compute(
        self,
        features: List[ft.FeatureBase],
    ) -> pd.DataFrame:
        builder = DuckDBBuilder(Path(self.config.engine_path))
        self.build_dataframes(builder)
        db = builder.db
        index_name = builder.index_name
        index = builder.index
        time_columns = None
        cutoff_time_table_name = None
        cutoff_time_col_name = None
        has_cutoff_time = self.config.use_cutoff_time and builder.cutoff_time is not None
        if has_cutoff_time:
            time_columns = builder.time_columns
            cutoff_time_table_name = builder.cutoff_time_table_name
            cutoff_time_col_name = builder.cutoff_time_col_name
        features = self.filter_nested_array_agg_features(features)
        logger.debug(
            f"Features to compute after filtering nested array agg: {pprint.pformat(features)}"
        )
        logger.debug("Generating SQLs ...")
        sqls = features2sql(
            features,
            index_name,
            has_cutoff_time=has_cutoff_time,
            cutoff_time_table_name=cutoff_time_table_name,
            cutoff_time_col_name=cutoff_time_col_name,
            time_col_mapping=time_columns,
        )
        sql_str = [format_sql(sql.sql()) for sql in sqls]
        logger.debug("Executing SQLs ...")
        dataframes = []
        for sql in tqdm.tqdm(sqls):
            logger.debug(f"Executing SQL: {format_sql(sql.sql())}")
            result = db.sql(sql.sql())
            if result is not None:
                dataframe = result.df()
                
                # drop the time column
                if cutoff_time_col_name in dataframe.columns:
                    dataframe.drop(columns=[cutoff_time_col_name], inplace=True)
                dataframe.rename(
                    decode_column_from_sql,
                    axis="columns",
                    inplace=True,
                )
                self.handle_array_aggregation(dataframe)
                dataframes.append(dataframe)
        logger.debug("Finalizing ...")
        ret_df = pd.DataFrame(
            reduce(lambda left, right: pd.merge(left, right, on=index_name), dataframes)
        )
        ret_df = ret_df.set_index(index_name).reindex(index).reset_index(drop=True)
        return ret_df

    def handle_array_aggregation(self, df: pd.DataFrame):
        array_agg_func_names = ["ARRAYMAX", "ARRAYMIN", "ARRAYMEAN"]
        for col in df.columns:
            num_array_agg = _check_array_agg_occurrences(col, array_agg_func_names)
            if num_array_agg == 1:
                if "ARRAYMAX" in col:
                    df[col] = df[col].apply(array_max)
                elif "ARRAYMIN" in col:
                    df[col] = df[col].apply(array_min)
                elif "ARRAYMEAN" in col:
                    df[col] = df[col].apply(array_mean)
            elif num_array_agg > 1:
                raise ValueError("The nested array aggregation has not supported yet")
        return df


def _check_array_agg_occurrences(col_name, array_agg_func_names) -> int:
    arr_agg_counts = 0
    for func_name in array_agg_func_names:
        arr_agg_counts += col_name.count(func_name)
    return arr_agg_counts


def _nanstack(arr_list: List[List]) -> np.ndarray:
    """Stack a list of numpy ndarrays that may contain NaN."""
    if arr_list is None:
        return np.nan  # all values are NaN
    arr_len = None
    for arr in arr_list:
        if isinstance(arr, List):
            arr_len = len(arr)
            break
    if arr_len is None:
        return np.nan  # all values are NaN
    fill_val = np.zeros(arr_len)
    new_arr_list = [arr if isinstance(arr, List) else fill_val for arr in arr_list]
    return np.stack(new_arr_list)


def array_max(column):
    if not isinstance(column, List):
        return np.nan
    stack = _nanstack(column)
    return stack.max(0) if isinstance(stack, np.ndarray) else np.nan


def array_min(column):
    if not isinstance(column, List):
        return np.nan
    stack = _nanstack(column)
    return stack.min(0) if isinstance(stack, np.ndarray) else np.nan


def array_mean(column):
    if not isinstance(column, List):
        return np.nan
    stack = _nanstack(column)
    return stack.mean(0) if isinstance(stack, np.ndarray) else np.nan
