import featuretools as ft
from sql_formatter.core import format_sql
from sqlglot.expressions import (
    Column,
    Table,
    Alias,
    Anonymous,
    Subquery,
    Join,
    Group,
    select,
    EQ,
    LT,
    Drop,
    Coalesce,
    Select,
    Identifier,
    Cast,
)
import sqlglot
from collections import defaultdict
from typing import Tuple, Dict, Optional, List
from enum import Enum
import pandas as pd
from loguru import logger


class JoinDirection(Enum):
    FORWARD = 1
    BACKWARD = 2
    SKIP_BACKWARD = 3


def encode_column_for_sql(name) -> str:
    replaced_string = name.replace("(", "{")  # raplace ( to {
    replaced_string = replaced_string.replace(")", "}")  # raplace ) to }
    replaced_string = replaced_string.replace(".", "-")  # replace . to -
    return replaced_string  # add "``"


def decode_column_from_sql(name) -> str:
    replaced_string = name.replace("{", "(")  # raplace { to (
    replaced_string = replaced_string.replace("}", ")")  # raplace } to )
    replaced_string = replaced_string.replace("-", ".")  # replace - to .
    return replaced_string


def str_to_quoted_identifier(name: str) -> Optional[Identifier]:
    if name is None:
        return None
    return Identifier(this=name, quoted=True)


class FeatureBlock:
    """A Feature block recursively builds a SQL AST for a featuretools feature."""

    def __init__(
        self,
        feature: ft.Feature,
        key_to_join_parent: Optional[str] = None,
        group_by_primitive: Optional[ft.primitives.base.PrimitiveBase] = None,
        generated_column_name: Optional[str] = None,
        skip_backward_relationships: Optional[
            ft.entityset.relationship.Relationship
        ] = None,
        is_first: Optional[bool] = False,
        index_col: Optional[str] = None,
        column_type_map: Optional[Dict[Tuple[str, str], str]] = None,
    ):
        """
        1. initialize the info from ft.Feature
        """
        self._feature = feature
        self._target_table_name = feature.dataframe_name
        self._primitive = feature.primitive
        self._key_to_join_parent = key_to_join_parent
        self._group_by_primitive = group_by_primitive
        # Descendant group by primitive is the first group by primitive in the descendant feature block.
        self._descendant_group_by_primitive = group_by_primitive
        self._child = None
        self._is_first = is_first
        self._generated_column_name = generated_column_name
        self._skip_backward_relationships = skip_backward_relationships
        self._index_col = index_col
        self._use_cutoff_time = False
        self._column_type_map = column_type_map or {}
        self._build()

        if group_by_primitive is None and self._child is not None:
            self._descendant_group_by_primitive = (
                self._child._descendant_group_by_primitive
            )

    def _build(self):
        assert (
            len(self._feature.base_features) <= 1
        ), "Only support at most one relationship path."
        if len(self._feature.base_features) > 0:
            # Recursively build SQL for the child.
            _, child_join_key = self._get_join_keys()
            child_skip_backward_relationships = (
                self._get_child_skip_backward_relationships()
            )
            child_group_by_primitive = None
            child_generated_column_name = None
            if isinstance(self._primitive, ft.primitives.base.AggregationPrimitive):
                child_group_by_primitive = self._primitive
                child_generated_column_name = encode_column_for_sql(
                    self._feature.get_name()
                )
            """
            2. Recursively build child featureblock
            """
            self._child = type(self)(
                self._feature.base_features[0],
                key_to_join_parent=child_join_key,
                group_by_primitive=child_group_by_primitive,
                generated_column_name=child_generated_column_name,
                skip_backward_relationships=child_skip_backward_relationships,
                is_first=False,
                column_type_map=self._column_type_map,
            )
        """
            3.get alias table name after building child featureblock
        """
        self._alias_table_name = self._generate_alias_table_name()
        if self._child is None:
            self._target_column_name = encode_column_for_sql(self._feature.get_name())
        else:
            self._target_column_name = self._child._generated_column_name
        if self._generated_column_name is None:
            self._generated_column_name = self._target_column_name

    def has_skip_backward_relationships(self) -> bool:
        if self._skip_backward_relationships is not None:
            return True
        if self._child is None:
            return False
        return self._child.has_skip_backward_relationships()

    def _get_join_keys(self) -> Tuple[str, str]:
        """Get the join keys for the target table and the child table."""
        assert len(self._feature.base_features) > 0
        relationships_with_directions = (
            self._feature.relationship_path._relationships_with_direction
        )
        join_keys = (
            relationships_with_directions[0][1]._parent_column_name,
            relationships_with_directions[0][1]._child_column_name,
        )
        if relationships_with_directions[0][0]:
            # If it is a forward relationship, then the join keys are reversed.
            join_keys = join_keys[::-1]
        return join_keys

    def _get_join_direction(self) -> Optional[JoinDirection]:
        """Get the join direction with the child feature.

        Return:
            - None if there is no relationship path.
            - JoinDirection.FORWARD if there is only one forward relationship.
            - JoinDirection.BACKWARD if there is only one backward relationship.
            - JoinDirection.SKIP_BACKWARD if there is more than one relationship.
        """
        if len(self._feature.base_features) == 0:
            return None
        relationships_with_directions = (
            self._feature.relationship_path._relationships_with_direction
        )
        if len(relationships_with_directions) > 1:
            return JoinDirection.SKIP_BACKWARD
        if relationships_with_directions[0][0]:
            return JoinDirection.FORWARD
        return JoinDirection.BACKWARD

    def _get_child_skip_backward_relationships(
        self,
    ) -> Optional[List[Tuple[bool, ft.entityset.relationship.Relationship]]]:
        """Get the child skip backward relationships."""
        relationships_with_directions = (
            self._feature.relationship_path._relationships_with_direction
        )
        if len(relationships_with_directions) <= 1:
            return None
        return relationships_with_directions[1:]

    def _generate_alias_table_name(self) -> str:
        direction = self._get_join_direction()
        target_table_name = self._target_table_name
        if direction is None:
            return target_table_name + "_alias"
        connection = "FORWARD" if direction == JoinDirection.FORWARD else "BACKWARD"
        name = f"{target_table_name}_{connection}_{self._child._alias_table_name}"
        if self._skip_backward_relationships is not None:
            name = "_SKIPBW_".join(
                [r[1]._parent_dataframe_name for r in self._skip_backward_relationships]
                + [name]
            )
        return name

    def _get_table_to_join_parent(self) -> str:
        """Get the table to join parent.

        If there is no skip backward relationship, then the table to join parent is the target table.
        Otherwise, then the table to join parent is the parent table of the first backward relationship.
        """
        if not self._use_cutoff_time:
            if self._skip_backward_relationships is None:
                return self._target_table_name
            else:
                return self._skip_backward_relationships[0][1]._parent_dataframe_name

    def _get_target_table_name(self) -> str:
        if not self._use_cutoff_time:
            return self._target_table_name

    def _get_child_table_name(self) -> str:
        assert self._child is not None
        return self._child._alias_table_name

    def _get_columns_to_select(self, table_name=None) -> List[Column]:
        """Get the columns to select from the target table.

        The columns to select include the key to join parent and the generated column.
        """
        table_to_join_parent = self._get_table_to_join_parent()
        if table_name is None:
            target_table_name = self._get_target_table_name()
        else:
            target_table_name = table_name
        columns = []
        if self._key_to_join_parent is not None:
            columns.append(
                Column(
                    this=str_to_quoted_identifier(self._key_to_join_parent),
                    table=str_to_quoted_identifier(table_to_join_parent),
                )
            )
        is_aggregation = self._group_by_primitive is not None
        if self._child is not None:
            assert self._child._alias_table_name is not None
            source_column = Column(
                this=str_to_quoted_identifier(self._child._generated_column_name),
                table=str_to_quoted_identifier(self._child._alias_table_name),
            )
        else:
            source_column = Column(
                this=str_to_quoted_identifier(self._target_column_name),
                table=str_to_quoted_identifier(target_table_name),
            )

        def _coalesce(col):
            # If next aggregation primitive is count, then we need to coalesce the column to 0.
            if (
                self._descendant_group_by_primitive is None
                or self._descendant_group_by_primitive.name != "count"
            ):
                return col, None
            return Coalesce(this=col, expressions="0"), col.this

        if is_aggregation:
            generated_column = self.handle_agg(source_column)
        else:
            generated_column = source_column
        generated_column, coalesce_alias = _coalesce(generated_column)
        if is_aggregation or self._is_first or coalesce_alias is not None:
            if is_aggregation or self._is_first:
                alias = self._generated_column_name
            else:
                alias = coalesce_alias
            generated_column = Alias(
                this=generated_column,
                alias=str_to_quoted_identifier(alias),
            )
        columns.append(generated_column)
        if self._is_first and self._index_col is not None:
            columns.append(
                Column(
                    this=str_to_quoted_identifier(self._index_col),
                    table=str_to_quoted_identifier(target_table_name),
                )
            )
        return columns

    def _get_group_by_columns(self) -> List[Column]:
        table_to_join_parent = self._get_table_to_join_parent()
        return [
            Column(
                this=str_to_quoted_identifier(self._key_to_join_parent),
                table=str_to_quoted_identifier(table_to_join_parent),
            )
        ]

    def _get_group_by_cond(self) -> Group:
        """Get the group by condition."""
        if self._group_by_primitive is None:
            return None
        return Group(
            expressions=self._get_group_by_columns(),
        )

    def _get_child_join_condition(self) -> EQ:
        target_join_key, child_join_key = self._get_join_keys()
        target_table_name = self._get_target_table_name()
        child_table_name = this = self._child._alias_table_name
        return EQ(
            this=Column(
                this=str_to_quoted_identifier(target_join_key),
                table=str_to_quoted_identifier(target_table_name),
            ),
            expression=Column(
                this=str_to_quoted_identifier(child_join_key),
                table=str_to_quoted_identifier(child_table_name),
            ),
        )

    def _get_joins(self, depth: int) -> List[Join]:
        """Get the group by condition."""
        joins = []
        if self._child is not None:
            source_table = Table(
                this=str_to_quoted_identifier(self._child._alias_table_name)
            )
            source_sql = Subquery(this=self._child.gen_sql(depth), alias=source_table)
            condition = self._get_child_join_condition()
            join_subquery = Join(
                this=source_sql,
                kind="left",
                on=condition,
            )
            joins.append(join_subquery)
        if self._skip_backward_relationships is not None:
            for _, relationship in self._skip_backward_relationships:
                skip_join = Join(
                    this=Table(
                        this=str_to_quoted_identifier(
                            relationship._parent_dataframe_name
                        )
                    ),
                    kind="right",
                    on=EQ(
                        this=Column(
                            this=str_to_quoted_identifier(
                                relationship._parent_column_name
                            ),
                            table=str_to_quoted_identifier(
                                relationship._parent_dataframe_name
                            ),
                        ),
                        expression=Column(
                            this=str_to_quoted_identifier(
                                relationship._child_column_name
                            ),
                            table=str_to_quoted_identifier(
                                relationship._child_dataframe_name
                            ),
                        ),
                    ),
                )
                joins.append(skip_join)
        return joins

    def gen_sql(self, depth: int = 0) -> Select:
        target_table = Table(
            this=str_to_quoted_identifier(self._get_target_table_name())
        )
        # 1.select a.parent FK and b.feature column c.index col
        columns = self._get_columns_to_select()
        # 2.LEFT JOIN children subquery
        joins = self._get_joins(depth + 1)
        # 3. get the groupby part
        group_by_cond = self._get_group_by_cond()
        # 4. build the select, join and groupby parts
        ast = select(*columns).from_(target_table)
        for j in joins:
            ast = ast.join(j)
        if group_by_cond is not None:
            ast = ast.group_by(group_by_cond)
        return ast

    def _get_base_identity_feature(self, feature):
        """Recursively find the base IdentityFeature from a feature.
        
        Args:
            feature: A featuretools Feature object
            
        Returns:
            The base IdentityFeature, or None if not found
        """
        # If it's an IdentityFeature, we've found it
        if isinstance(feature, ft.IdentityFeature):
            return feature
        
        # If it's an AggregationFeature or DirectFeature, check base features
        if isinstance(feature, (ft.AggregationFeature, ft.DirectFeature)):
            if hasattr(feature, 'base_features') and len(feature.base_features) > 0:
                return self._get_base_identity_feature(feature.base_features[0])
        
        return None

    def _is_source_feature_boolean(self) -> bool:
        """Check if the source feature being aggregated is boolean type.
        
        Returns:
            True if the source feature has boolean variable_type, False otherwise.
        """
        # Get the source feature being aggregated
        if self._child is not None:
            # If there's a child, the source feature is the child feature
            source_feature = self._child._feature
        else:
            # If no child, the source feature is the base feature
            if len(self._feature.base_features) > 0:
                source_feature = self._feature.base_features[0]
            else:
                # Direct feature - check the feature itself
                source_feature = self._feature
        
        # Try to find the base IdentityFeature for more reliable type checking
        base_identity = self._get_base_identity_feature(source_feature)
        if base_identity is not None:
            source_feature = base_identity
        
        # Method 0: Check column type map (most reliable if available)
        if hasattr(source_feature, 'dataframe_name') and hasattr(source_feature, 'get_name'):
            dataframe_name = source_feature.dataframe_name
            column_name = source_feature.get_name()
            key = (dataframe_name, column_name)
            if key in self._column_type_map:
                dtype_str = str(self._column_type_map[key]).lower()
                # Check for boolean dtypes
                if 'bool' in dtype_str or dtype_str in ['boolean', 'bool']:
                    return True

        return False

    def handle_agg(self, source_column) -> Anonymous:
        # Aggregations that require numeric types (don't work with boolean in DuckDB)
        numeric_only_primitives = ["mean", "std", "sum"]
        
        array_agg_func_names = ["arraymax", "arraymin", "arraymean"]
        if self._group_by_primitive.name in array_agg_func_names:
            return Anonymous(this="array_agg", expressions=[source_column])
        elif self._group_by_primitive.name == "join":
            return Anonymous(this="string_agg", expressions=[source_column, "'\n'"])
        elif self._group_by_primitive.name in numeric_only_primitives:
            # Only cast boolean to integer for numeric aggregations
            # Check if source feature is boolean before casting
            if self._is_source_feature_boolean():
                cast_expression = Cast(
                    this=source_column,
                    to=Identifier(this="INTEGER", quoted=False)
                )
            else:
                # If not boolean, use the source column directly
                cast_expression = source_column
            # Map primitives to backend-specific function names (DuckDB)
            # Use sample standard deviation to match pandas/featuretools default semantics (ddof=1)
            func_name = (
                "stddev_samp" if self._group_by_primitive.name == "std" else self._group_by_primitive.name
            )
            return Anonymous(this=func_name, expressions=[cast_expression])
        else:
            return Anonymous(
                this=self._group_by_primitive.name, expressions=[source_column]
            )


class FeatureBlockWithCutoffTime(FeatureBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_cutoff_time = True
        self._temp_time_table_name = None
        self._time_col = None

    def _get_table_to_join_parent(self) -> str:
        assert (
            self._skip_backward_relationships is None
        ), "Not implemented skip backward with cutoff time yet."
        return self._temp_time_table_name

    def _get_target_table_name(self) -> str:
        return self._temp_time_table_name

    def _get_child_join_condition(self) -> EQ:
        condition = super()._get_child_join_condition()
        target_table_name = self._get_target_table_name()
        child_table_name = self._child._alias_table_name
        return condition.and_(
            EQ(
                this=Column(
                    this=str_to_quoted_identifier(self._time_col),
                    table=str_to_quoted_identifier(target_table_name),
                ),
                expression=Column(
                    this=str_to_quoted_identifier(self._child._time_col),
                    table=str_to_quoted_identifier(child_table_name),
                ),
            )
        )

    def _get_columns_to_select(self) -> List[Column]:
        target_table_name = self._get_target_table_name()
        columns = super()._get_columns_to_select(target_table_name)
        columns.append(
            Column(
                this=str_to_quoted_identifier(self._time_col),
                table=str_to_quoted_identifier(target_table_name),
            )
        )
        return columns

    def _get_group_by_columns(self) -> List[Column]:
        columns = super()._get_group_by_columns()
        table_to_join_parent = self._get_table_to_join_parent()
        columns.append(
            Column(
                this=str_to_quoted_identifier(self._time_col),
                table=str_to_quoted_identifier(table_to_join_parent),
            )
        )
        return columns

    def _get_intermediate_columns_to_select(
        self, parent_temp_time_table, time_col
    ) -> List[Column]:
        columns = []
        columns.append(
            Column(
                this=str_to_quoted_identifier(time_col),
                table=str_to_quoted_identifier(parent_temp_time_table),
            )
        )
        target_table_name = self._target_table_name
        key_to_join_parent = self._key_to_join_parent
        if key_to_join_parent is not None:
            columns.append(
                Column(
                    this=str_to_quoted_identifier(key_to_join_parent),
                    table=str_to_quoted_identifier(target_table_name),
                )
            )
        if self._child is not None:
            this_join_key, _ = self._get_join_keys()
            columns.append(
                Column(
                    this=str_to_quoted_identifier(this_join_key),
                    table=str_to_quoted_identifier(target_table_name),
                )
            )
        else:
            columns.append(
                Column(
                    this=str_to_quoted_identifier(self._target_column_name),
                    table=str_to_quoted_identifier(target_table_name),
                )
            )
        # Unique columns
        columns = list(set(columns))
        return columns

    def _get_cutoff_time_joins(
        self, parent_temp_time_table, parent_join_key, parent_time_col, time_col_mapping
    ) -> List[Join]:
        joins = []
        target_table_name = self._target_table_name
        # 1. Join PK and FK
        condition = EQ(
            this=Column(
                this=str_to_quoted_identifier(parent_join_key),
                table=str_to_quoted_identifier(parent_temp_time_table),
            ),
            expression=Column(
                this=str_to_quoted_identifier(self._key_to_join_parent),
                table=str_to_quoted_identifier(target_table_name),
            ),
        )
        # 2. Temporal constraint
        if target_table_name in time_col_mapping:
            condition = condition.and_(
                LT(
                    this=Column(
                        this=str_to_quoted_identifier(
                            time_col_mapping[target_table_name]
                        ),
                        table=str_to_quoted_identifier(target_table_name),
                    ),
                    expression=Column(
                        this=str_to_quoted_identifier(parent_time_col),
                        table=str_to_quoted_identifier(parent_temp_time_table),
                    ),
                )
            )
        joins.append(
            Join(
                this=Table(this=str_to_quoted_identifier(target_table_name)),
                kind="left",
                on=condition,
            )
        )
        return joins

    def gen_intermediate_timestamp_sql(
        self,
        parent_temp_time_table: str,
        parent_join_key: str,
        parent_time_col: str,
        parent_join_direction: JoinDirection,
        time_col_mapping: Dict[str, str],
        depth: int = 1,
    ):
        self._time_col = parent_time_col
        columns = self._get_intermediate_columns_to_select(
            parent_temp_time_table,
            parent_time_col,
        )
        from_table = Table(this=str_to_quoted_identifier(parent_temp_time_table))
        joins = self._get_cutoff_time_joins(
            parent_temp_time_table, parent_join_key, parent_time_col, time_col_mapping
        )
        ast = select(*columns).from_(from_table)
        for j in joins:
            ast = ast.join(j)
        temp_time_table_name = f"{self._feature.dataframe_name}_TEMP_TIME_DEPTH_{depth}"
        if parent_join_direction == JoinDirection.FORWARD:
            ast = ast.distinct()
        ast = ast.ctas(temp_time_table_name)
        asts = [ast]
        temp_tables = [temp_time_table_name]
        self._temp_time_table_name = temp_time_table_name
        if self._child is not None:
            this_join_key, _ = self._get_join_keys()
            child_asts, child_temp_tables = self._child.gen_intermediate_timestamp_sql(
                parent_temp_time_table=temp_time_table_name,
                parent_join_key=this_join_key,
                parent_time_col=self._time_col,
                parent_join_direction=self._get_join_direction(),
                time_col_mapping=time_col_mapping,
                depth=depth + 1,
            )
            asts.extend(child_asts)
            temp_tables.extend(child_temp_tables)
        return asts, temp_tables


def gen_target_temp_time_table(
    target_table_name: str,
    cutoff_time_table_name: str,
    cutoff_time_col_name: str,
    index_col_name:str
):
    temp_table_name = f"{target_table_name}_TEMP_TIME_DEPTH_0"
    ast = (
        select(
            f"{target_table_name}.*",
            Column(
                this=str_to_quoted_identifier(cutoff_time_col_name),
                table=str_to_quoted_identifier(cutoff_time_table_name),
            ),
        )
        .from_(Table(this=str_to_quoted_identifier(target_table_name)))
        .join(
            Join(
                this=Table(this=str_to_quoted_identifier(cutoff_time_table_name)),
                kind="left",
                on=EQ(
                    this=Column(
                        this=str_to_quoted_identifier(index_col_name),
                        table=str_to_quoted_identifier(target_table_name),
                    ),
                    expression=Column(
                        this=str_to_quoted_identifier(index_col_name),
                        table=str_to_quoted_identifier(cutoff_time_table_name),
                    ),
                ),
            )
        )
        .ctas(temp_table_name)
    )
    return ast, temp_table_name


def features2sql(
    features: List[ft.Feature],
    index_name: str,
    has_cutoff_time: bool,
    cutoff_time_table_name: str,
    cutoff_time_col_name: str,
    time_col_mapping: Dict[str, str],
    column_type_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Select]:
    if has_cutoff_time:
        target_table = features[0].dataframe_name
        target_time_ast, temp_target_table_name = gen_target_temp_time_table(
            target_table,
            cutoff_time_table_name,
            cutoff_time_col_name,
            index_name,
        )
        ret_asts = []
        for f in features:
            feature_block = FeatureBlockWithCutoffTime(
                f,
                generated_column_name=encode_column_for_sql(f.get_name()),
                is_first=True,
                index_col=index_name,
                column_type_map=column_type_map,
            )
            if feature_block.has_skip_backward_relationships():
                logger.warning(
                    "Skip backward relationships are not supported with cutoff time yet."
                )
                continue
            feature_block._time_col = cutoff_time_col_name
            feature_block._temp_time_table_name = temp_target_table_name
            asts = [target_time_ast]
            temp_tables = [temp_target_table_name]
            if feature_block._child is not None:
                this_join_key, _ = feature_block._get_join_keys()
                (
                    child_asts,
                    child_temp_tables,
                ) = feature_block._child.gen_intermediate_timestamp_sql(
                    parent_temp_time_table=temp_target_table_name,
                    parent_join_key=this_join_key,
                    parent_time_col=cutoff_time_col_name,
                    parent_join_direction=feature_block._get_join_direction(),
                    time_col_mapping=time_col_mapping,
                )
                asts.extend(child_asts)
                temp_tables.extend(child_temp_tables)
            sql = feature_block.gen_sql()
            asts.append(sql)
            for temp_table in temp_tables:
                asts.append(Drop(this=temp_table, kind="table"))
            ret_asts.extend(asts)
        return ret_asts
    sqls = []
    for f in features:
        feature_block = FeatureBlock(
            f,
            generated_column_name=encode_column_for_sql(f.get_name()),
            is_first=True,
            index_col=index_name,
            column_type_map=column_type_map,
        )
        sql = feature_block.gen_sql()
        logger.debug(f"Generated SQL for {f.get_name()}: \n" f"{format_sql(sql.sql())}")
        sqls.append(sql)
    sqls = group_sqls(sqls)
    return sqls


def group_sqls(sqls: List[Select]) -> List[Select]:
    new_sqls = []
    grouped_sqls = defaultdict(list)
    for i, sql in enumerate(sqls):
        grouped_sqls[get_join_path(sql)].append((i, sql))

    for group in grouped_sqls.values():
        idxs = [g[0] for g in group]
        merging_sqls = [g[1] for g in group]
        merged_sql = merge(merging_sqls)
        new_sqls.append(merged_sql)
        if len(idxs) > 1:
            logger.debug(f"Merge features with indexes {idxs} into one sql.")
            sqls_str = "\n".join([format_sql(sql.sql()) for sql in merging_sqls])
            logger.debug("Merging SQLs: \n" f"{sqls_str}")
            logger.debug("Merged SQL: \n" f"{format_sql(merged_sql.sql())}")
    return new_sqls


def get_join_path(sql) -> str:
    ret = ()
    if isinstance(sql, Select):
        ret += ("SELECT", sql.args["from"].sql())
        if "joins" in sql.args:
            for join in sql.args["joins"]:
                ret += get_join_path(join)
        if "group" in sql.args:
            ret += get_join_path(sql.args["group"])
    elif isinstance(sql, Join):
        ret += ("JOIN", sql.args["kind"], sql.args["on"].sql())
        ret += get_join_path(sql.args["this"])
    elif isinstance(sql, Group):
        ret += ("GROUP", sql.sql())
    elif isinstance(sql, Subquery):
        ret += get_join_path(sql.args["this"])
        if "alias" in sql.args:
            ret += ("AS", sql.args["alias"].sql())
    else:
        ret += (sql.sql(),)
    return ret


def merge(sqls) -> Select:
    if len(sqls) == 1:
        return sqls[0]
    new_sql = sqls[0].copy()

    if isinstance(new_sql, Select):
        exps = [sql.args["expressions"] for sql in sqls]
        exps_set = set(exp for sublist in exps for exp in sublist)
        new_sql.set("expressions", exps_set)
        if "joins" in new_sql.args:
            new_joins = []
            for i in range(len(new_sql.args["joins"])):
                new_joins.append(merge([sql.args["joins"][i] for sql in sqls]))
            new_sql.set("joins", new_joins)
    elif isinstance(new_sql, Join):
        new_sql.set("this", merge([sql.args["this"] for sql in sqls]))
    elif isinstance(new_sql, Subquery):
        new_sql.set("this", merge([sql.args["this"] for sql in sqls]))

    return new_sql
