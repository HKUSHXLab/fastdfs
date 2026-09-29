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
from woodwork.column_schema import ColumnSchema

from ..dataset.rdb import RDB
from ..dataset.meta import RDBColumnDType, RDBColumnSchema

__all__ = [
    'DFSConfig',
    'DFSEngine',
    'dfs_feature_column_name',
    'get_dfs_engine',
    'dfs_engine',
    'Quantile25',
    'Quantile75',
    'DiscreteEntropy',
]


def dfs_feature_column_name(feature: ft.FeatureBase) -> str:
    """Stable column name for a DFS feature, including Featuretools relationship depth.

    Depth is ``FeatureBase.get_depth()`` (0 = identity on the target row, 1 = one hop, …).
    Format: ``d{depth}__`` + ``feature.get_name()`` so the original Featuretools name stays
    parseable after the prefix.

    Examples
    --------
    - ``d0__intime`` — target-frame identity / time column feature at depth 0
    - ``d2__patients.MEAN(admissions.MEAN(icustays.los_icu))`` — nested aggregation at depth 2
    """
    return f"d{int(feature.get_depth())}__{feature.get_name()}"


# Custom quantile aggregation primitives
class Quantile25(ft.primitives.AggregationPrimitive):
    """Calculates the 25th percentile (Q1) of a numeric column."""
    name = "quantile_25"
    input_types = [ColumnSchema(semantic_tags=['numeric'])]
    return_type = ColumnSchema(semantic_tags=['numeric'])
    
    def get_function(self):
        def quantile_25(x):
            return x.quantile(0.25)
        return quantile_25


class Quantile75(ft.primitives.AggregationPrimitive):
    """Calculates the 75th percentile (Q3) of a numeric column."""
    name = "quantile_75"
    input_types = [ColumnSchema(semantic_tags=['numeric'])]
    return_type = ColumnSchema(semantic_tags=['numeric'])
    
    def get_function(self):
        def quantile_75(x):
            return x.quantile(0.75)
        return quantile_75


class DiscreteEntropy(ft.primitives.AggregationPrimitive):
    """Calculates the discrete entropy of a categorical column.
    
    Entropy formula: -Σ(p * log2(p)) where p is the probability of each category.
    This measures the uncertainty/randomness in the categorical distribution.
    """
    name = "discrete_entropy"
    input_types = [ColumnSchema(semantic_tags=['category'])]
    return_type = ColumnSchema(semantic_tags=['numeric'])
    
    def get_function(self):
        def discrete_entropy(x):
            from collections import Counter
            import numpy as np
            # Drop NaN/None values
            non_null_values = x.dropna()
            if len(non_null_values) == 0:
                return np.nan
            # Count frequencies
            counts = Counter(non_null_values)
            total = len(non_null_values)
            # Calculate probabilities
            probs = [count / total for count in counts.values()]
            # Calculate entropy: -Σ(p * log2(p))
            return -sum(p * np.log2(p) for p in probs if p > 0)
        return discrete_entropy

class DFSConfig(pydantic.BaseModel):
    """
    Configuration model for Deep Feature Synthesis parameters.

    This class defines all the configurable parameters for the DFS process,
    including aggregation primitives, depth limits, and engine selection.

    Attributes:
        agg_primitives: List of aggregation primitive names to use. See
            ``docs/aggregation_primitives.md`` in the package repo for supported
            primitives per engine (dfs2sql vs featuretools).
        max_depth: Maximum depth for feature generation
        use_cutoff_time: Whether to use temporal cutoff times
        engine: Name of the DFS engine to use for computation
        engine_path: Optional path for engine-specific configuration
        max_features: Maximum number of features to generate
        include_entities: List of entities to include in feature generation
        ignore_entities: List of entities to ignore in feature generation
        chunk_size: Chunk size for batch processing
        n_jobs: Number of parallel jobs for computation
        dfs2sql_concat_chunk_size: dfs2sql only — number of per-feature SQL frames to concat per
            intermediate block when assembling the wide matrix (default 512).
        dfs2sql_sql_workers: Number of concurrent queries sharing one DuckDB database.
        dfs2sql_merge_queries: Merge compatible feature projections before execution.
        dfs2sql_threads: Global DuckDB thread count, independent of worker count.
        dfs2sql_memory_limit: DuckDB memory limit; does not limit total process memory.
        dfs2sql_temp_directory: Directory for DuckDB spill files.
        include_cutoff_time: When True, include RDB rows whose time equals the task cutoff
            (``<=`` in dfs2sql; featuretools ``include_cutoff_time=True``). Default False keeps
            strict ``<`` for other datasets.
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
    engine: str = "dfs2sql"
    engine_path: Optional[str] = None
    max_features: int = -1
    include_entities: Optional[List[str]] = None
    ignore_entities: Optional[List[str]] = None
    chunk_size: Optional[int] = None
    n_jobs: int = 1
    # Generate "traverse-back" features that featuretools' DFS skips. Featuretools
    # refuses to revisit a table already on the synthesis path, so it misses features
    # whose path returns to an earlier table through a different relationship edge
    # (e.g. a graph nodes <- edges(src_id, dst_id): nodes -> edges -> nodes -> edges,
    # or a bipartite users -> interactions -> items -> interactions -> users). These
    # revisiting paths collect genuinely different rows and yield valid multi-hop /
    # neighborhood aggregations. Set False to disable (recommended on dense or wide
    # multi-FK schemas where the feature count can explode).
    traverse_back: bool = True
    # Maximum number of table revisits allowed along a single traverse-back feature
    # path. 1 permits one return to an already-visited table (e.g. A->B->A->B or
    # A->B->C->B->A); higher values permit deeper neighborhood walks at the cost of
    # combinatorial blow-up. Bounded in practice by max_depth as well.
    max_revisits: int = 1
    # dfs2sql: when stitching one DuckDB result per feature, concat this many skinny frames
    # at a time before a final horizontal concat (memory vs overhead tradeoff).
    dfs2sql_concat_chunk_size: int = 512
    # Concurrent queries share one DuckDB database.
    dfs2sql_sql_workers: int = 1
    dfs2sql_merge_queries: bool = True
    dfs2sql_threads: Optional[int] = pydantic.Field(default=None, ge=1)
    dfs2sql_memory_limit: Optional[str] = None
    dfs2sql_temp_directory: Optional[str] = None
    include_cutoff_time: bool = False


class DFSEngine:
    """Base class for DFS computation engines."""

    def __init__(self, config: DFSConfig):
        self.config = config

    def compute_features(
        self,
        rdb: RDB,
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
        rdb: RDB,
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

        target_index = "__target_index__"
        if target_index in target_dataframe.columns:
            logger.error(f"Target dataframe cannot contain reserved column name '{target_index}'.")
        # Assign creates a new dataframe but reuses underlying blocks where possible.
        target_df_with_index = target_dataframe.assign(
            **{target_index: np.arange(len(target_dataframe))}
        )

        # Optimization: Filter target dataframe to only include necessary columns
        # This defines the contract for the engines: they only see keys, time, and index.
        columns_to_keep = {target_index}
        columns_to_keep.update(key_mappings.keys())
        if cutoff_time_column:
            columns_to_keep.add(cutoff_time_column)
            
        # Work on a defensive copy that engines are free to mutate.
        target_df_for_engine = target_df_with_index[list(columns_to_keep)].copy()

        # Phase 1: Feature preparation (common logic in base class)
        features = self.prepare_features(rdb, target_df_for_engine, key_mappings, cutoff_time_column, config)

        if len(features) == 0:
            logger.warning("No features generated, check your configuration or data.")
            return target_dataframe

        # Phase 2: Feature computation (engine-specific logic in subclasses)
        feature_matrix = self.compute_feature_matrix(
            rdb, target_df_for_engine, key_mappings, cutoff_time_column, features, config
        )

        renames = {}
        for f in features:
            old = f.get_name()
            new = dfs_feature_column_name(f)
            if old != new and old in feature_matrix.columns:
                renames[old] = new
        if renames:
            feature_matrix = feature_matrix.rename(columns=renames, copy=False)

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
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> List[ft.FeatureBase]:
        """
        Prepare feature specifications using featuretools DFS.

        This method builds the EntitySet, runs featuretools DFS to generate
        feature specifications, and filters the results based on configuration.
        
        For multiple key mappings, processes each separately to work around
        featuretools' path-sharing bug where shared downstream entities omit
        feature generation from secondary relationship paths.

        Returns:
            List of featuretools FeatureBase objects representing features to compute
        """
        
        # Workaround for multiple keys: process each key separately
        if len(key_mappings) > 1:
            logger.info(f"Detected {len(key_mappings)} key mappings. Processing each separately to avoid path conflicts.")
            all_features = []
            feature_names_seen = set()
            
            for key_idx, (key_col, rdb_ref) in enumerate(key_mappings.items(), 1):
                logger.info(f"Processing key {key_idx}/{len(key_mappings)}: {key_col} -> {rdb_ref}")
                
                # Recursively call with single key
                single_key_features = self.prepare_features(
                    rdb, target_dataframe, {key_col: rdb_ref}, cutoff_time_column, config
                )
                
                # Collect unique features
                new_count = 0
                for feat in single_key_features:
                    feat_name = feat.get_name()
                    if feat_name not in feature_names_seen:
                        all_features.append(feat)
                        feature_names_seen.add(feat_name)
                        new_count += 1
                
                logger.info(f"Added {new_count} unique features from {key_col}")
            
            logger.info(f"Total unique features from all keys: {len(all_features)}")
            return all_features
        
        # Single key mapping - standard DFS logic
        entity_set = self._build_entity_set_from_rdb(rdb)

        target_entity_name = "__target__"
        target_index = "__target_index__"

        entity_set = entity_set.add_dataframe(
            dataframe_name=target_entity_name,
            dataframe=target_dataframe,
            index=target_index,
            time_index=cutoff_time_column
        )

        self._add_target_relationships(entity_set, target_entity_name, key_mappings)

        logger.debug(entity_set)

        agg_primitives = self._convert_primitives(config.agg_primitives)

        dfs_kwargs = {
            'entityset': entity_set,
            'target_dataframe_name': target_entity_name,
            'max_depth': config.max_depth,
            'agg_primitives': agg_primitives,
            'trans_primitives': [],
            'features_only': True
        }

        if config.max_features > 0:
            dfs_kwargs['max_features'] = config.max_features
        if config.include_entities:
            dfs_kwargs['include_entities'] = config.include_entities
        if config.ignore_entities:
            dfs_kwargs['ignore_entities'] = config.ignore_entities

        features = ft.dfs(**dfs_kwargs)

        filtered_features = self._filter_features(features, entity_set, target_entity_name, config)

        # Generate traverse-back features that featuretools misses
        traverse_back_features = self._generate_traverse_back_features(
            filtered_features, entity_set, target_entity_name, key_mappings, config
        )
        if traverse_back_features:
            logger.info(f"Generated {len(traverse_back_features)} traverse-back features")
            filtered_features.extend(traverse_back_features)

        return filtered_features

    def _build_entity_set_from_rdb(self, rdb: RDB) -> ft.EntitySet:
        """Build EntitySet from RDB tables only (adapted from existing build_dataframes logic)."""

        entity_set = ft.EntitySet(id=rdb.metadata.name)

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
        # Skip self-referential relationships to avoid recursion errors
        for child_table, child_col, parent_table, parent_col in rdb.get_relationships():
            # Skip self-referential relationships (e.g., posts.ParentId -> posts.Id)
            if child_table == parent_table:
                logger.warning(f"Skipping self-referential relationship: {child_table}.{child_col} -> {parent_table}.{parent_col}")
                continue
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
        """Convert primitive names to primitive objects.
        
        Supports:
        - Built-in featuretools primitives (by name string)
        - Custom quantile primitives: 'quantile_25', 'quantile_75'
        - Custom discrete entropy primitive: 'discrete_entropy'
        """
        primitives = []
        for prim in primitive_names:
            # Handle custom quantile primitives
            if prim == "quantile_25":
                primitives.append(Quantile25())
            elif prim == "quantile_75":
                primitives.append(Quantile75())
            elif prim == "discrete_entropy":
                primitives.append(DiscreteEntropy())
            else:
                # Built-in featuretools primitives (passed as string name)
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

    def _generate_traverse_back_features(
        self,
        existing_features: List[ft.FeatureBase],
        entity_set: ft.EntitySet,
        target_entity_name: str,
        key_mappings: Dict[str, str],
        config: DFSConfig,
    ) -> List[ft.FeatureBase]:
        """
        Generate "traverse-back" features that featuretools' DFS skips.

        Featuretools enumerates only *table-simple* synthesis paths: its
        ``_run_dfs`` guard (``if dataframe in all_features: continue``) refuses to
        re-enter a table already expanded on the active branch. It therefore misses
        every feature whose path revisits an earlier table, even when the revisit
        collects genuinely different rows. Two important shapes are missed:

          * homogeneous graph -- ``A -> B -> A -> B`` where ``B`` has >= 2 FKs into
            the entity table ``A`` (e.g. ``nodes <- edges(src_id, dst_id)``);
          * bipartite / heterogeneous -- ``A -> B -> C -> B -> A`` (e.g.
            ``users -> interactions -> items -> interactions -> users``, the classic
            collaborative-filtering shape).

        This method enumerates these features directly as a bounded, non-degenerate
        walk over the schema's FK multigraph and materializes each walk into an
        ordinary featuretools ``FeatureBase`` (so both engines compute them, cutoff
        times included).

        General rule (see dev_logs/022_TRAVERSE_BACK_FEATURES_PLAN.md, section 10):

          * Walk from the target along FK edges in either direction (``up`` =
            DirectFeature/join, ``down`` = AggregationFeature).
          * Length bounded by ``max_depth`` (featuretools ``get_depth()``).
          * Table revisits bounded by ``max_revisits``.
          * **Degeneracy prune**: never wrap a feature in an ``AggregationFeature``
            down edge ``e`` when its immediate base is a ``DirectFeature`` up the
            *same* edge ``e`` -- that round trip is the identity (a many->one climb
            followed by re-grouping by the same parent recovers the original group).
          * Aggregation hops are type-matched to the inner feature's output schema.

        Only features that actually revisit a table are returned; plain
        table-simple features (already produced by featuretools) are filtered out by
        name against ``existing_features``.
        """
        if not config.traverse_back:
            return []
        if config.max_depth < 4:
            # A revisiting path needs at least 4 hops (target -> A -> B -> A -> B).
            return []

        from featuretools.feature_base import AggregationFeature, DirectFeature, IdentityFeature
        from featuretools.entityset.relationship import RelationshipPath
        from woodwork.column_schema import ColumnSchema

        max_depth = config.max_depth
        max_revisits = max(0, config.max_revisits)
        if max_revisits == 0:
            return []
        # A single conceptual traverse-back "round" (A -> B -> A -> B) re-enters two
        # tables (A and B again), so the raw table-revisit budget is 2 per round.
        revisit_budget = 2 * max_revisits

        # ------------------------------------------------------------------ #
        # Relationship key columns (to identify keys and avoid aggregating them).
        # ------------------------------------------------------------------ #
        key_columns = set()
        for rel in entity_set.relationships:
            key_columns.add((rel._parent_dataframe_name, rel._parent_column_name))
            key_columns.add((rel._child_dataframe_name, rel._child_column_name))

        # ------------------------------------------------------------------ #
        # Adjacency on the FK multigraph: table -> list of (relationship, direction,
        # neighbor_table). 'up' follows a child->parent FK (DirectFeature); 'down'
        # follows parent->child (AggregationFeature).
        # ------------------------------------------------------------------ #
        def neighbors(table):
            out = []
            for rel in entity_set.relationships:
                if rel._child_dataframe_name == table:
                    out.append((rel, "up", rel._parent_dataframe_name))
                if rel._parent_dataframe_name == table:
                    out.append((rel, "down", rel._child_dataframe_name))
            # deterministic order
            out.sort(key=lambda x: (x[1], x[0]._child_dataframe_name,
                                    x[0]._child_column_name,
                                    x[0]._parent_dataframe_name))
            return out

        # ------------------------------------------------------------------ #
        # Enumerate target-rooted walks that revisit at least one table, bounded by
        # depth and max_revisits. Each walk is a list of (relationship, direction,
        # next_table). We never walk back into the target table.
        # ------------------------------------------------------------------ #
        walks: List[List[Tuple[Any, str, str]]] = []

        def walk(table, path, n_revisits, visited):
            if len(path) >= max_depth:
                return
            for rel, direction, nb in neighbors(table):
                if nb == target_entity_name:
                    continue
                revisited = nb in visited
                nr = n_revisits + (1 if revisited else 0)
                if nr > revisit_budget:
                    continue
                new_path = path + [(rel, direction, nb)]
                # only keep walks that revisit at least one table (the rest are
                # plain table-simple features featuretools already generated)
                if nr >= 1:
                    walks.append(new_path)
                walk(nb, new_path, nr, visited | {nb})

        walk(target_entity_name, [], 0, {target_entity_name})

        if not walks:
            return []

        # ------------------------------------------------------------------ #
        # Aggregation primitives, resolved to objects.
        # ------------------------------------------------------------------ #
        resolved_primitives = []
        for prim in self._convert_primitives(config.agg_primitives):
            if isinstance(prim, str):
                prim_cls = ft.primitives.utils.get_aggregation_primitives().get(prim)
                if prim_cls is not None:
                    resolved_primitives.append(prim_cls())
            else:
                resolved_primitives.append(prim)

        def _tags(column_schema):
            if isinstance(column_schema, ColumnSchema):
                return set(column_schema.semantic_tags or set())
            return set()

        count_prims = [p for p in resolved_primitives
                       if (getattr(p, "name", "") or "").lower() == "count"]
        value_prims = [p for p in resolved_primitives
                       if (getattr(p, "name", "") or "").lower() != "count"]

        def _agg_prims_for(column_schema):
            """Aggregation primitives whose input type matches a feature's output."""
            seed_tags = _tags(column_schema)
            out = []
            for prim in value_prims:
                input_types = getattr(prim, "input_types", None)
                if not input_types:
                    continue
                in_tags = _tags(input_types[0])
                if in_tags and seed_tags and (in_tags & seed_tags):
                    out.append(prim)
            return out

        # ------------------------------------------------------------------ #
        # Seed features on a table: identity features of its non-key columns.
        # ------------------------------------------------------------------ #
        def _seed_features(table):
            df = entity_set[table]
            seeds = []
            for col in df.columns:
                if (table, col) in key_columns:
                    continue
                tags = set(df.ww.semantic_tags.get(col, set()))
                if "index" in tags or "foreign_key" in tags:
                    continue
                seeds.append(IdentityFeature(df.ww[col]))
            return seeds

        # ------------------------------------------------------------------ #
        # Materialize a walk into target-rooted features, inside-out.
        # ------------------------------------------------------------------ #
        existing_names = {f.get_name() for f in existing_features}
        new_features: List[ft.FeatureBase] = []
        seen_new = set()

        def _emit(feature):
            if feature.get_depth() > max_depth:
                return
            name = feature.get_name()
            if name in existing_names or name in seen_new:
                return
            if config.max_features > 0 and (
                len(existing_names) + len(seen_new) >= config.max_features
            ):
                return
            new_features.append(feature)
            seen_new.add(name)

        def _materialize(path):
            deepest = path[-1][2]
            # The innermost hop being 'down' means we can also seed with a COUNT over
            # the deepest table's index; otherwise seed with identity value features.
            feats = list(_seed_features(deepest))
            innermost_rel, innermost_dir, _ = path[-1]

            # Build inside-out: process hops from deepest to the target.
            for i in range(len(path) - 1, -1, -1):
                rel, direction, table = path[i]
                parent_or_child = target_entity_name if i == 0 else path[i - 1][2]
                next_feats = []
                if direction == "down":
                    # AggregationFeature: parent_or_child <- table via rel
                    for f in feats:
                        # Degeneracy prune: never aggregate (down e) a feature whose
                        # immediate base is a DirectFeature up the SAME edge e.
                        if isinstance(f, DirectFeature):
                            try:
                                _, f_rel = f.relationship_path[0]
                            except (IndexError, TypeError):
                                f_rel = None
                            if f_rel is rel:
                                continue
                        prims = _agg_prims_for(f.column_schema)
                        for prim in prims:
                            try:
                                next_feats.append(AggregationFeature(
                                    [f], parent_or_child, prim,
                                    RelationshipPath([(False, rel)]),
                                ))
                            except (AssertionError, Exception):
                                continue
                    # COUNT(table) seed: the innermost down hop can count rows.
                    if i == len(path) - 1:
                        for cprim in count_prims:
                            try:
                                idx_feat = IdentityFeature(
                                    entity_set[table].ww[entity_set[table].ww.index]
                                )
                                next_feats.append(AggregationFeature(
                                    [idx_feat], parent_or_child, cprim,
                                    RelationshipPath([(False, rel)]),
                                ))
                            except (AssertionError, Exception):
                                continue
                else:
                    # DirectFeature: parent_or_child (child) -> table (parent) via rel
                    for f in feats:
                        try:
                            next_feats.append(DirectFeature(f, parent_or_child, rel))
                        except (AssertionError, Exception):
                            continue
                feats = next_feats
                if not feats:
                    break

            for f in feats:
                if f.dataframe_name == target_entity_name:
                    _emit(f)

        for path in walks:
            if config.max_features > 0 and (
                len(existing_names) + len(seen_new) >= config.max_features
            ):
                break
            _materialize(path)

        return new_features


    @abc.abstractmethod
    def compute_feature_matrix(
        self,
        rdb: RDB,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        features: List[ft.FeatureBase],
        config: DFSConfig
    ) -> pd.DataFrame:
        """
        Compute actual feature values from feature specifications.

        This is engine-specific logic implemented by subclasses.

        Contract:
        1. Input `target_dataframe`:
           - Contains ONLY the necessary columns:
             - `__target_index__`: Unique identifier for alignment.
             - Columns specified in `key_mappings`.
             - `cutoff_time_column` (if provided).
           - Does NOT contain other columns from the original user input.
        
        2. Output DataFrame:
           - Must contain `__target_index__` column (or index) for alignment.
           - Must contain the computed feature columns.
           - Should NOT contain columns from the input `target_dataframe` (keys, cutoff time)
             to avoid duplication when merging back to the original data.
           - Must return a valid DataFrame even if no features are computed (containing just the index).
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
        log_ty = "NaturalLanguage"
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

def get_primitive_name(feature):
    """Extract primitive name from an AggregationFeature."""
    if isinstance(feature, ft.AggregationFeature):
        primitive = feature.primitive
        if hasattr(primitive, 'name'):
            return primitive.name.lower()
        elif hasattr(primitive, '__name__'):
            return primitive.__name__.lower()
        elif isinstance(primitive, str):
            return primitive.lower()
    return None

def base_feature_is_key(feature, keys):
    """
    Check if a feature's base feature is a key (primary/foreign key).
    
    Exception: COUNT over an entity is valid even if it uses keys internally,
    because it counts records (rows), not aggregating key values.
    
    For COUNT(entity), the structure is:
    - AggregationFeature(COUNT) -> IdentityFeature(entity.index)
    - This counts rows, not key values, so it's valid and should NOT be filtered.
    """
    # Special case: COUNT over entire entity (not aggregating a key column value)
    # COUNT(entity) uses the entity's index (a key) for grouping, but counts rows
    if isinstance(feature, ft.AggregationFeature):
        if get_primitive_name(feature) == 'count':
            # COUNT over entity: base is IdentityFeature of entity's index
            if feature.base_features:
                base = feature.base_features[0]
                if isinstance(base, ft.IdentityFeature):
                    # This is COUNT(entity) - counts rows, not key values
                    # Even though it uses a key internally, it's valid
                    return False
    
    # For DirectFeature wrapping COUNT(entity), also allow it
    if isinstance(feature, ft.DirectFeature):
        if feature.base_features:
            base = feature.base_features[0]
            # Check if base is COUNT over entity
            if isinstance(base, ft.AggregationFeature):
                if get_primitive_name(base) == 'count':
                    if base.base_features:
                        agg_base = base.base_features[0]
                        if isinstance(agg_base, ft.IdentityFeature):
                            # COUNT over entity - don't filter
                            return False
            # Otherwise check recursively
            return base_feature_is_key(base, keys)
    
    # Original logic: recursively check base features
    if isinstance(feature, (ft.AggregationFeature, ft.DirectFeature)):
        if feature.base_features:
            return base_feature_is_key(feature.base_features[0], keys)
    
    # Check if IdentityFeature is a key
    elif isinstance(feature, ft.IdentityFeature):
        return (feature.dataframe_name, feature.get_name()) in keys
    
    else:
        raise NotImplementedError(f'Unsupported subfeature {feature}')
