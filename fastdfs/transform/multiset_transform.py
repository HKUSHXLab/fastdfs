"""
Multiset feature extraction transform.

This module implements FeaturizeMultiset, a ColumnTransform that extracts 
features from multiset columns (columns containing lists of labels).

Example:
transform_pipeline = RDBTransformPipeline([
    HandleDummyTable(),
    FillMissingPrimaryKey(),
    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "hour"])),
    RDBTransformWrapper(FeaturizeMultiset(top_n=20, retain_original=False)),
    RDBTransformWrapper(FilterColumn(drop_dtypes=["text"])),
    RDBTransformWrapper(CanonicalizeTypes())
])
"""

from typing import List, Tuple
import pandas as pd
from collections import Counter
from ..dataset.meta import RDBColumnSchema, RDBColumnDType
from .base import ColumnTransform


class FeaturizeMultiset(ColumnTransform):
    """Extract features from multiset columns (lists of labels)."""
    
    def __init__(self, top_n: int = 20, retain_original: bool = True):
        """
        Initialize multiset featurizer.
        
        Args:
            top_n: Number of top labels to create binary columns for. Default is 20.
            retain_original: Whether to keep the original multiset column. Default is True.
        """
        self.top_n = top_n
        self.retain_original = retain_original
    
    def applies_to(self, column_metadata: RDBColumnSchema) -> bool:
        """Check if this transform should be applied to multiset columns."""
        return column_metadata.dtype == RDBColumnDType.multiset_t
    
    def __call__(self, column: pd.Series, column_metadata: RDBColumnSchema) -> Tuple[pd.DataFrame, List[RDBColumnSchema]]:
        """Extract features from a multiset column."""
        
        # Check if this is a multiset column
        if column_metadata.dtype != RDBColumnDType.multiset_t:
            # Return original column as DataFrame
            return pd.DataFrame({column_metadata.name: column}), [column_metadata]
        
        base_name = column_metadata.name
        
        # Step 1: Collect all labels from all rows
        all_labels = []
        for value in column:
            # Skip None values
            if value is None:
                continue
            
            # Check type first to avoid pd.isna ambiguity with arrays
            if isinstance(value, (list, tuple)):
                # For lists/tuples, check for None/NaN elements
                if len(value) == 0:
                    continue
                all_labels.extend(value)
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                # Handle numpy arrays or other iterables
                try:
                    # Convert to list first
                    value_list = list(value)
                    if len(value_list) == 0:
                        continue
                    all_labels.extend(value_list)
                except (TypeError, ValueError):
                    # Skip if conversion fails
                    continue
            else:
                # Scalar value - check for NaN
                try:
                    if pd.isna(value):
                        continue
                except (ValueError, TypeError):
                    # If pd.isna fails, skip this value
                    continue
        
        # Step 2: Find top-n labels by frequency
        label_counter = Counter(all_labels)
        top_labels = [label for label, _ in label_counter.most_common(self.top_n)]
        
        # If fewer than top_n labels exist, use all available labels
        # (This is already handled by most_common which returns all if fewer than n)
        
        # Step 3: Create binary columns for top-n labels
        feature_df = pd.DataFrame(index=column.index)
        new_column_schemas = []
        
        def label_in_multiset(multiset_value, target_label, true_value="1", false_value="0"):
            """Check if target_label is in the multiset_value."""
            # Skip None values
            if multiset_value is None:
                return false_value
            
            # Check type first to avoid pd.isna ambiguity with arrays
            if isinstance(multiset_value, (list, tuple)):
                return true_value if target_label in multiset_value else false_value
            elif hasattr(multiset_value, '__iter__') and not isinstance(multiset_value, str):
                try:
                    return true_value if target_label in list(multiset_value) else false_value
                except (TypeError, ValueError):
                    return false_value
            else:
                # Scalar value - check for NaN
                try:
                    if pd.isna(multiset_value):
                        return false_value
                except (ValueError, TypeError):
                    # If pd.isna fails, treat as non-matching
                    return false_value
                return false_value
        
        for label in top_labels:
            col_name = f"{base_name}_has_{label}"
            # Create binary indicator: "1" if label is in the row's multiset, "0" otherwise
            #feature_df[col_name] = column.apply(lambda x: label_in_multiset(x, label))
            feature_df[col_name] = column.apply(lambda x: label_in_multiset(x, label))#, true_value=1, false_value=0))
            
            new_col_schema = RDBColumnSchema(
                name=col_name,
                dtype=RDBColumnDType.category_t,
                #dtype=RDBColumnDType.float_t
            )
            new_column_schemas.append(new_col_schema)
        
        # Step 4: Create others_count column
        others_count_col_name = f"{base_name}_others_count"
        
        def count_others(multiset_value):
            """Count labels not in top_n."""
            # Skip None values
            if multiset_value is None:
                return 0.0
            
            # Check type first to avoid pd.isna ambiguity with arrays
            if isinstance(multiset_value, (list, tuple)):
                labels = multiset_value
            elif hasattr(multiset_value, '__iter__') and not isinstance(multiset_value, str):
                try:
                    labels = list(multiset_value)
                except (TypeError, ValueError):
                    return 0.0
            else:
                # Scalar value - check for NaN
                try:
                    if pd.isna(multiset_value):
                        return 0.0
                except (ValueError, TypeError):
                    # If pd.isna fails, treat as empty
                    return 0.0
                return 0.0
            
            # Count labels that are not in top_labels
            others = [label for label in labels if label not in top_labels]
            #return float(len(others))
            return "1" if len(others) > 0 else "0"
        
        feature_df[others_count_col_name] = column.apply(count_others)
        
        others_count_schema = RDBColumnSchema(
            name=others_count_col_name,
            #dtype=RDBColumnDType.float_t
            dtype=RDBColumnDType.category_t
        )
        new_column_schemas.append(others_count_schema)
        
        # Step 5: Handle original column retention. Make sure that no multiset type is passed forward.
        if self.retain_original:
            feature_df[base_name] = column
            new_column_schemas.insert(0, RDBColumnSchema(name=base_name, dtype=RDBColumnDType.text_t))
        
        return feature_df, new_column_schemas
