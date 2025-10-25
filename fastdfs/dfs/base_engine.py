"""
New DFS Engine Interface for table-centric feature engineering.

This module implements the new DFS engine interface that works with external
target dataframes and simplified RDB datasets, removing the dependency on tasks.
"""

import abc
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import featuretools as ft
import numpy as np
from loguru import logger
import pydantic

from ..dataset.rdb import RDBDataset
from ..dataset.meta import RDBColumnDType, RDBColumnSchema

__all__ = ['DFSConfig', 'DFSEngine', 'get_dfs_engine', 'dfs_engine']

class DFSConfig(pydantic.BaseModel):
    """
    Configuration model for Deep Feature Synthesis parameters.

    This class defines all the configurable parameters for the DFS process,
    including aggregation primitives, depth limits, and engine selection.

    Attributes:
        agg_primitives: List of aggregation primitive names to use
        max_depth: Maximum depth for feature generation
        use_cutoff_time: Whether to use temporal cutoff times
        engine: Name of the DFS engine to use for computation
        engine_path: Optional path for engine-specific configuration
        trans_primitives: List of transformation primitives to use
        where_primitives: List of where primitives to use
        max_features: Maximum number of features to generate
        include_entities: List of entities to include in feature generation
        ignore_entities: List of entities to ignore in feature generation
        chunk_size: Chunk size for batch processing
        n_jobs: Number of parallel jobs for computation
    """
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "std",
        "count",
        "mode",
    ]
    max_depth: int = 2
    use_cutoff_time: bool = True
    engine: str = "featuretools"
    engine_path: Optional[str] = "/tmp/duck.db"
    trans_primitives: List[str] = []
    where_primitives: List[str] = []
    max_features: int = -1
    include_entities: Optional[List[str]] = None
    ignore_entities: Optional[List[str]] = None
    chunk_size: Optional[int] = None
    n_jobs: int = 1


class DFSEngine:
    """Base class for DFS computation engines."""

    def __init__(self, config: DFSConfig):
        self.config = config

    def compute_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute DFS features for a target dataframe using RDB context.

        Args:
            rdb: The relational database providing context for feature generation
            target_dataframe: DataFrame to augment with features (doesn't need to exist in RDB)
            key_mappings: Map from target_dataframe columns to RDB primary keys
                         e.g., {"user_id": "user.user_id", "item_id": "item.item_id"}
            cutoff_time_column: Column name in target_dataframe for temporal cutoff (optional)
            config_overrides: Dictionary of config parameters to override for this computation

        Returns:
            DataFrame with original target_dataframe data plus generated features
        """
        # Merge config overrides
        effective_config = self.config.copy(deep=True)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(effective_config, key):
                    setattr(effective_config, key, value)

        return self._compute_features_impl(
            rdb, target_dataframe, key_mappings, cutoff_time_column, effective_config
        )

    def _compute_features_impl(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Implementation-specific feature computation."""

        # Handle empty target dataframe
        if len(target_dataframe) == 0:
            return target_dataframe

        original_index = target_dataframe.index

        # Add default index to target dataframe before prepare_features
        target_df_with_index = target_dataframe.copy()
        target_index = "__target_index__"
        if target_index in target_df_with_index.columns:
            logger.error(f"Target dataframe cannot contain reserved column name '{target_index}'.")
        target_df_with_index[target_index] = range(len(target_df_with_index))

        # Create a working copy for engine-specific preparation steps so that
        # featuretools/woodwork mutations do not affect the merge copy that
        # preserves the original target order.
        target_df_for_engine = target_df_with_index.copy()

        # Phase 1: Feature preparation (common logic in base class)
        features = self.prepare_features(rdb, target_df_for_engine, key_mappings, cutoff_time_column, config)

        if len(features) == 0:
            logger.warning("No features generated, check your configuration or data.")
            return target_dataframe

        # Phase 2: Feature computation (engine-specific logic in subclasses)
        feature_matrix = self.compute_feature_matrix(
            rdb, target_df_for_engine, key_mappings, cutoff_time_column, features, config
        )

        if len(feature_matrix.columns) != len(features):
            logger.error("Feature matrix column count does not match prepared features.")

        if target_index not in feature_matrix.columns:
            logger.error("Feature matrix is missing the target index column '__target_index__'.")

        # Align features with original target rows using the target index
        target_with_idx = target_df_with_index.set_index(target_index)
        features_with_idx = feature_matrix.set_index(target_index)

        merged = target_with_idx.join(features_with_idx, how="left")

        # Restore the original index ordering
        merged.index = original_index

        return merged

    def prepare_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> List[ft.FeatureBase]:
        """
        Prepare feature specifications using featuretools DFS.

        This method builds the EntitySet, runs featuretools DFS to generate
        feature specifications, and filters the results based on configuration.
        This is common logic shared by all engines.

        Returns:
            List of featuretools FeatureBase objects representing features to compute
        """
        # Build EntitySet from RDB tables
        entity_set = self._build_entity_set_from_rdb(rdb)

        # Add target dataframe as temporary entity
        target_entity_name = "__target__"
        target_index = "__target_index__"  # This should already be in the dataframe

        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_dataframe,
            index=target_index,
            time_index=cutoff_time_column
        )

        # Add relationships from target to RDB entities
        self._add_target_relationships(entity_set, target_entity_name, key_mappings)

        logger.debug(entity_set)

        # Convert primitive names to objects
        agg_primitives = self._convert_primitives(config.agg_primitives)

        # Generate feature specifications using featuretools
        dfs_kwargs = {
            'entityset': entity_set,
            'target_dataframe_name': target_entity_name,
            'max_depth': config.max_depth,
            'agg_primitives': agg_primitives,
            'trans_primitives': config.trans_primitives,
            'features_only': True
        }

        # Add optional parameters if specified
        if config.max_features > 0:
            dfs_kwargs['max_features'] = config.max_features
        if config.include_entities:
            dfs_kwargs['include_entities'] = config.include_entities
        if config.ignore_entities:
            dfs_kwargs['ignore_entities'] = config.ignore_entities

        features = ft.dfs(**dfs_kwargs)

        # Filter features based on configuration
        filtered_features = self._filter_features(features, entity_set, target_entity_name, config)

        return filtered_features

    def _build_entity_set_from_rdb(self, rdb: RDBDataset) -> ft.EntitySet:
        """Build EntitySet from RDB tables only (adapted from existing build_dataframes logic)."""

        entity_set = ft.EntitySet(id=rdb.metadata.dataset_name)

        # Add all RDB tables as entities
        for table_name in rdb.table_names:
            df = rdb.get_table(table_name)
            table_meta = rdb.get_table_metadata(table_name)

            # Parse columns and build logical types/semantic tags (reuse existing logic)
            logical_types = {}
            semantic_tags = {}
            index_col = None

            for col_schema in table_meta.columns:
                col_name = col_schema.name
                col_data = df[col_name].values
                series, log_ty, tag = parse_one_column(col_schema, col_data)
                logical_types[col_name] = log_ty

                if col_schema.dtype == RDBColumnDType.primary_key:
                    index_col = col_name
                    # Don't set semantic tag for index
                else:
                    semantic_tags[col_name] = tag

            # Add default index if needed
            if index_col is None:
                df["__index__"] = np.arange(len(df))
                index_col = "__index__"

            entity_set = entity_set.add_dataframe(
                dataframe_name=table_name,
                dataframe=df,
                index=index_col,
                time_index=table_meta.time_column,
                logical_types=logical_types,
                semantic_tags=semantic_tags
            )

        # Add relationships between RDB tables
        for child_table, child_col, parent_table, parent_col in rdb.get_relationships():
            entity_set = entity_set.add_relationship(
                parent_dataframe_name=parent_table,
                parent_column_name=parent_col,
                child_dataframe_name=child_table,
                child_column_name=child_col
            )

        return entity_set

    def _add_target_relationships(
        self, entity_set: ft.EntitySet, target_entity_name: str, key_mappings: Dict[str, str]
    ):
        """Add relationships from target entity to RDB entities."""

        for target_col, rdb_ref in key_mappings.items():
            parent_table, parent_col = rdb_ref.split('.')

            entity_set = entity_set.add_relationship(
                parent_dataframe_name=parent_table,
                parent_column_name=parent_col,
                child_dataframe_name=target_entity_name,
                child_column_name=target_col
            )

    def _convert_primitives(self, primitive_names: List[str]) -> List:
        """Convert primitive names to primitive objects (simplified, no array primitives)."""
        primitives = []
        for prim in primitive_names:
            # Only support basic primitives, no array types
            primitives.append(prim)
        return primitives

    def _filter_features(
        self,
        features: List[ft.FeatureBase],
        entity_set: ft.EntitySet,
        target_entity_name: str,
        config: DFSConfig
    ) -> List[ft.FeatureBase]:
        """Filter features (adapted from existing filter_features logic)."""

        if len(features) == 0:
            return features

        # Get foreign/primary keys from relationships
        keys = set()
        for rel in entity_set.relationships:
            keys.add((rel.parent_name, rel.parent_column.name))
            keys.add((rel.child_name, rel.child_column.name))

        new_features = []
        for feat in features:
            feat_str = str(feat)

            # Remove features involving the target table
            if target_entity_name in feat_str:
                continue

            # Remove key-based features
            if base_feature_is_key(feat, keys):
                continue

            new_features.append(feat)

        return new_features

    @abc.abstractmethod
    def compute_feature_matrix(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """
        Compute actual feature values from feature specifications.

        This is engine-specific logic implemented by subclasses.
        """
        pass


# Registry for DFS engines (reuse existing pattern)
_DFS_ENGINE_REGISTRY = {}

def dfs_engine(engine_class):
    """Decorator to register DFS engines."""
    _DFS_ENGINE_REGISTRY[engine_class.name] = engine_class
    return engine_class

def get_dfs_engine(name: str, config: DFSConfig) -> DFSEngine:
    """Get DFS engine by name."""
    if name not in _DFS_ENGINE_REGISTRY:
        raise ValueError(f"Unknown DFS engine: {name}")
    return _DFS_ENGINE_REGISTRY[name](config)

# Internal utilities

def parse_one_column(
    col_schema: RDBColumnSchema, col_data: np.ndarray
) -> Tuple[pd.Series, str, str]:
    if col_schema.dtype == RDBColumnDType.category_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "category"
    elif col_schema.dtype == RDBColumnDType.float_t:
        if col_data.ndim > 1:
            series = pd.Series(list(col_data))
            log_ty = "Array"
            tag = "array"
        else:
            series = pd.Series(col_data, copy=False)
            log_ty = "Double"
            tag = "numeric"
    elif col_schema.dtype == RDBColumnDType.datetime_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Datetime"
        tag = "string"
    elif col_schema.dtype == RDBColumnDType.text_t:
        series = pd.Series(col_data, copy=False)
        log_ty = "Text"
        tag = "text"
    elif col_schema.dtype == RDBColumnDType.primary_key:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "index"
    elif col_schema.dtype == RDBColumnDType.foreign_key:
        series = pd.Series(col_data, copy=False)
        log_ty = "Categorical"
        tag = "foreign_key"
    else:
        raise ValueError(f"Unsupported dtype {col_schema.dtype}.")
    return series, log_ty, tag

def base_feature_is_key(feature, keys):
    if isinstance(feature, (ft.AggregationFeature, ft.DirectFeature)):
        return base_feature_is_key(feature.base_features[0], keys)
    elif isinstance(feature, ft.IdentityFeature):
        return (feature.dataframe_name, feature.get_name()) in keys
    else:
        raise NotImplementedError(f'Unsupported subfeature {feature}')