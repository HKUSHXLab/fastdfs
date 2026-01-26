import pandas as pd
import numpy as np
from typing import Optional
from ..dataset.meta import RDBColumnDType

def safe_convert_to_string(series: pd.Series) -> pd.Series:
    """
    Safely convert a pandas Series to string type for key consistency.
    
    Raises:
         ValueError: If the series is of floating point type.
    """
    if series.empty:
        return series.astype(str)

    # Strictly forbid Float
    if pd.api.types.is_float_dtype(series):
        print(series)
        raise ValueError(f"Cannot safe convert float column to string key. Please convert {series.name} to integer or string explicitly.")

    return series.astype(str)

def _is_multiset_column(col_data: pd.Series) -> bool:
    """
    Check if a column contains multiset data (lists/arrays of strings).
    
    Args:
        col_data: pandas Series to check
        
    Returns:
        True if the column appears to contain multiset data (lists of strings)
    """
    # Must be object dtype to potentially contain lists
    if not pd.api.types.is_object_dtype(col_data):
        return False
    
    # Sample some non-null values to check
    non_null_data = col_data.dropna()
    if len(non_null_data) == 0:
        return False
    
    # Sample up to 100 values to check
    sample_size = min(100, len(non_null_data))
    sample = non_null_data.sample(n=sample_size, random_state=42) if len(non_null_data) > sample_size else non_null_data
    
    list_count = 0
    for value in sample:
        # Check if value is a list or array-like
        if isinstance(value, (list, tuple, np.ndarray)):
            # Check if all elements are strings
            if len(value) > 0:
                if all(isinstance(item, str) for item in value):
                    list_count += 1
                else:
                    # If any element is not a string, it's not a multiset
                    return False
            else:
                # Empty list could still be a multiset
                list_count += 1
        else:
            # If we find a non-list value, it's not a multiset
            # But allow some tolerance for mixed types (e.g., some NaN handling)
            if not pd.isna(value):
                return False
    
    # If most sampled values are lists of strings, consider it a multiset
    # Require at least 50% to be lists
    return list_count / len(sample) >= 0.5

def infer_semantic_type(
    series: pd.Series,
    col_name: str,
    pk_col: Optional[str] = None,
    time_col: Optional[str] = None,
    is_foreign_key: bool = False,
    explicit_type_hint: Optional[str] = None,
    category_threshold: int = 10
) -> RDBColumnDType:
    """
    Infer the semantic type of a column.
    
    Args:
        series: Pandas Series containing the data.
        col_name: Name of the column.
        pk_col: Name of the primary key column for the table.
        time_col: Name of the time column for the table.
        is_foreign_key: Whether the column is a foreign key.
        explicit_type_hint: Optional explicit type string.
        category_threshold: Threshold for unique values to consider categorical.
        
    Returns:
        RDBColumnDType
    """
    # Check PK first
    if col_name == pk_col:
        return RDBColumnDType.primary_key
        
    # Check FK next
    if is_foreign_key:
        return RDBColumnDType.foreign_key

    # Check hints
    if explicit_type_hint:
        return RDBColumnDType(explicit_type_hint)
        
    # Check Time Col
    if col_name == time_col:
        return RDBColumnDType.datetime_t
        
    # Infer from data
    if pd.api.types.is_datetime64_any_dtype(series):
        return RDBColumnDType.datetime_t
        
    if pd.api.types.is_float_dtype(series):
        return RDBColumnDType.float_t
        
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        # Treat integers as float for feature engineering usually
        return RDBColumnDType.float_t
    
    # Check for multiset (list/array of strings)
    if _is_multiset_column(series):
        return RDBColumnDType.multiset_t
        
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        try:
            n_unique = series.nunique()
            if n_unique < category_threshold:
                return RDBColumnDType.category_t
            else:
                return RDBColumnDType.text_t
        except TypeError:
            return RDBColumnDType.text_t
            
    return RDBColumnDType.text_t
