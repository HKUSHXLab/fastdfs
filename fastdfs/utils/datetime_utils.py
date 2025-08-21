"""Datetime utilities."""
import pandas as pd
import numpy as np
from typing import List

def dt2ts(dt : np.ndarray) -> np.ndarray:
    dt = dt.astype('datetime64[ns]')
    return (dt - np.array(0).astype('datetime64[ns]')).astype('int64')

def ts2dt(ts : np.ndarray) -> np.ndarray:
    return np.array(ts).astype('datetime64[ns]')

def dt2year(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.year.values

def dt2month(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.month.values

def dt2day(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.day.values

def dt2dayofweek(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.dayofweek.values

def dt2hour(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.hour.values

def dt2minute(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.minute.values

def dt2second(dt : np.ndarray) -> np.ndarray:
    dt_series = pd.to_datetime(pd.Series(dt))
    return dt_series.dt.second.values

def featurize_datetime_column(column: pd.Series, features: List[str]) -> pd.DataFrame:
    """
    Extract multiple datetime features from a datetime column.
    
    Args:
        column: Pandas Series with datetime values
        features: List of features to extract. Options: 
                 ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek']
    
    Returns:
        DataFrame with extracted datetime features
    """
    result_df = pd.DataFrame()
    base_name = column.name or 'datetime'
    
    # Map feature names to extraction functions
    feature_funcs = {
        'year': dt2year,
        'month': dt2month,
        'day': dt2day,
        'hour': dt2hour,
        'minute': dt2minute,
        'second': dt2second,
        'dayofweek': dt2dayofweek
    }
    
    # Extract requested features
    for feature in features:
        if feature in feature_funcs:
            feature_values = feature_funcs[feature](column.values)
            feature_name = f"{base_name}_{feature}"
            result_df[feature_name] = feature_values
    
    return result_df
