from typing import Tuple, Dict, Optional, List
from pathlib import Path
import pydantic
from loguru import logger
import numpy as np
from collections import defaultdict

from ..dataset import (
    DBBColumnDType,
    DBBRDBDataset,
    DBBRDBTask,
    DBBRDBTaskCreator,
    DBBRDBDatasetCreator,
    DBBTaskType,
)

from ..utils.device import DeviceInfo
from .base import RDBDatasetPreprocess, rdb_preprocess
from .transform import (
    RDBData,
    ColumnData,
    RDBTransform,
    get_rdb_transform_class,
    is_task_table,
    make_task_table_name,
    unmake_task_table_name,
)

class _NameAndConfig(pydantic.BaseModel):
    name : str
    config : Optional[Dict] = {}

class RDBTransformPreprocessConfig(pydantic.BaseModel):
    transforms : List[_NameAndConfig]

@rdb_preprocess
class RDBTransformPreprocess(RDBDatasetPreprocess):

    config_class = RDBTransformPreprocessConfig
    name : str = "transform"
    default_config = RDBTransformPreprocessConfig.parse_obj({
        "transforms" : [
            { "name" : "handle_dummy_table" },
            { "name" : "key_mapping" },
            {
                "name" : "column_transform_chain",
                "config" : {
                    "transforms": [
                        { "name" : "canonicalize_numeric" },
                        { "name" : "canonicalize_datetime" },
                        { "name" : "glove_text_embedding" },
                        {
                            "name" : "featurize_datetime",
                            "config" : {
                                "methods" : [
                                    "YEAR",
                                    "MONTH",
                                    "DAY",
                                    "DAYOFWEEK",
                                    "TIMESTAMP",
                                ],
                            },
                        },
                        { "name" : "norm_numeric" },
                        { "name" : "remap_category" },
                    ]
                },
            },
            { "name" : "filter_column" },
            { "name" : "fill_timestamp" },
        ],
    })

    def __init__(self, config : RDBTransformPreprocessConfig):
        super().__init__(config)
        self.transforms : List[RDBTransform] = []
        for xcfg in config.transforms:
            xform_class = get_rdb_transform_class(xcfg.name)
            xform_config = xform_class.config_class.parse_obj(xcfg.config)
            self.transforms.append(xform_class(xform_config))

    def run(
        self,
        dataset : DBBRDBDataset,
        output_path : Path,
        device : DeviceInfo
    ):
        logger.debug("Fit & transform RDB data.")
        rdb_data = self.extract_data(dataset)
        task_data_fit, task_data_transform = self.extract_task_data(dataset)
        all_data_fit = _merge_rdb_and_task(rdb_data, task_data_fit)
        for xform in self.transforms:
            all_data_fit = xform.fit_transform(all_data_fit, device)
        new_task_data_transform = {}
        for task_name, task_data in task_data_transform.items():
            logger.debug(f"Transform data of task {task_name}.")
            for xform in self.transforms:
                task_data = xform.transform(task_data, device)
            new_task_data_transform[task_name] = task_data
        new_rdb_data, new_task_data_fit = _split_rdb_and_task(all_data_fit)

        logger.debug(f"New RDB data:\n{new_rdb_data}")
        for task_name in new_task_data_fit:
            logger.debug(f"New task data ({task_name}):\n"
                         f"{new_task_data_fit[task_name]}\n"
                         f"{new_task_data_transform[task_name]}")

        new_name = dataset.metadata.dataset_name
        logger.debug(f"Generating new dataset {new_name}.")
        ctor = DBBRDBDatasetCreator(new_name)
        self.output_data(ctor, new_rdb_data)
        self.output_column_groups(ctor, new_rdb_data)
        self.output_tasks(ctor, new_task_data_fit, new_task_data_transform, dataset)
        ctor.done(output_path)

    def extract_data(self, dataset : DBBRDBDataset) -> RDBData:
        tables = {tbl_name : {} for tbl_name in dataset.tables}
        for tbl_schema in dataset.metadata.tables:
            tbl_name = tbl_schema.name
            for col_schema in tbl_schema.columns:
                col_name = col_schema.name
                col_meta = dict(col_schema)
                if col_name == tbl_schema.time_column:
                    col_meta['is_time_column'] = True
                tables[tbl_name][col_name] = ColumnData(
                    col_meta, dataset.tables[tbl_name][col_name])
        
        # Create column groups based on shared_schema
        column_groups = []
        
        # First, add existing column groups from dataset metadata
        if dataset.metadata.column_groups is not None:
            for cg in dataset.metadata.column_groups:
                column_groups.append([(cid.table, cid.column) for cid in cg])
        
        # Then, create column groups for task columns with shared_schema
        shared_schema_groups = defaultdict(list)
        for task in dataset.tasks:
            task_table_name = make_task_table_name(task.metadata.name)
            for col_schema in task.metadata.columns:
                if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
                    shared_schema = col_schema.shared_schema
                    # Add both the task column and the referenced data column to the group
                    task_col_ref = (task_table_name, col_schema.name)
                    data_table, data_col = shared_schema.split('.')
                    data_col_ref = (data_table, data_col)
                    
                    # Only add if not already in the group
                    if task_col_ref not in shared_schema_groups[shared_schema]:
                        shared_schema_groups[shared_schema].append(task_col_ref)
                    if data_col_ref not in shared_schema_groups[shared_schema]:
                        shared_schema_groups[shared_schema].append(data_col_ref)
        
        # Add the shared schema groups to column_groups
        for group in shared_schema_groups.values():
            if len(group) > 1:  # Only add groups with multiple columns
                column_groups.append(group)
        
        relationships = None
        if dataset.metadata.relationships is not None:
            relationships = []
            for rel in dataset.metadata.relationships:
                relationships.append((
                    rel.fk.table, rel.fk.column, rel.pk.table, rel.pk.column))
        return RDBData(tables, column_groups, relationships)

    def extract_task_data(
        self,
        dataset : DBBRDBDataset,
    ) -> Tuple[RDBData, Dict[str, RDBData]]:
        fit_table = {}
        transform_tables = {
            task.metadata.name : {}
            for task in dataset.tasks
        }
        
        # Build a mapping from table.column to schema for metadata inference
        data_schema_map = {}
        for tbl_schema in dataset.metadata.tables:
            for col_schema in tbl_schema.columns:
                key = f"{tbl_schema.name}.{col_schema.name}"
                data_schema_map[key] = col_schema
        
        for task_id, task in enumerate(dataset.tasks):
            task_name = task.metadata.name
            task_table_name = make_task_table_name(task_name)

            fit_table[task_table_name] = {}
            transform_tables[task_name][task_table_name] = {}
            for col_schema in task.metadata.columns:
                col = col_schema.name
                col_meta = dict(col_schema)
                
                # If column has shared_schema, infer metadata from the referenced data column
                if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
                    shared_schema_key = col_schema.shared_schema
                    if shared_schema_key in data_schema_map:
                        ref_schema = data_schema_map[shared_schema_key]
                        # Copy all metadata from referenced column, preserving explicitly set values
                        ref_meta_dict = dict(ref_schema)
                        for field, value in ref_meta_dict.items():
                            if field not in col_meta or col_meta[field] is None:
                                col_meta[field] = value
                
                if col == task.metadata.time_column:
                    col_meta['is_time_column'] = True
                
                # Training data always goes to fit_table for learning parameters
                train_data = task.train_set[col]
                fit_table[task_table_name][col] = ColumnData(col_meta, train_data)
                
                # Validation and test data go to transform_tables for applying learned transforms
                val_test_data = np.concatenate([
                    task.validation_set[col],
                    task.test_set[col]
                ], axis=0)
                transform_tables[task_name][task_table_name][col] = \
                    ColumnData(col_meta, val_test_data)
        
        # Create column groups for task data based on shared_schema
        task_column_groups = []
        for task in dataset.tasks:
            task_table_name = make_task_table_name(task.metadata.name)
            shared_schema_groups = defaultdict(list)
            for col_schema in task.metadata.columns:
                if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
                    shared_schema = col_schema.shared_schema
                    # Add both the task column and the referenced data column to the group
                    task_col_ref = (task_table_name, col_schema.name)
                    data_table, data_col = shared_schema.split('.')
                    data_col_ref = (data_table, data_col)
                    
                    # Only add if not already in the group
                    if task_col_ref not in shared_schema_groups[shared_schema]:
                        shared_schema_groups[shared_schema].append(task_col_ref)
                    if data_col_ref not in shared_schema_groups[shared_schema]:
                        shared_schema_groups[shared_schema].append(data_col_ref)

            # Add the shared schema groups to column_groups
            for group in shared_schema_groups.values():
                if len(group) > 1:  # Only add groups with multiple columns
                    task_column_groups.append(group)
            
        task_data_fit = RDBData(fit_table, task_column_groups, None)
        task_data_transform = {
            task_name : RDBData(task_table, task_column_groups, None)
            for task_name, task_table in transform_tables.items()
        }
        return task_data_fit, task_data_transform

    def output_data(self, ctor : DBBRDBDatasetCreator, rdb: RDBData):
        for tbl_name, table in rdb.tables.items():
            ctor.add_table(tbl_name)
            time_col = None
            for col_name, col in table.items():
                metadata = col.metadata
                if metadata.get('is_time_column', False):
                    metadata.pop('is_time_column')
                    time_col = col_name
                ctor.add_column(tbl_name, col_name, col.data, **metadata)
            if time_col is not None:
                ctor.set_time_column(tbl_name, time_col)

    def output_column_groups(self, ctor : DBBRDBDatasetCreator, rdb: RDBData):
        if rdb.column_groups is not None:
            for cg in rdb.column_groups:
                ctor.add_column_group(cg)

    def output_tasks(
        self,
        ctor : DBBRDBDatasetCreator,
        task_data_fit : Dict[str, RDBData],
        task_data_transform : Dict[str, RDBData],
        ds: DBBRDBDataset
    ):
        for orig_task in ds.tasks:
            task_name = orig_task.metadata.name
            task_table_name = make_task_table_name(task_name)
            all_data = {
                **task_data_fit[task_name].tables[task_table_name],
                **task_data_transform[task_name].tables[task_table_name]
            }
            num_train, num_val, num_test = _get_num(orig_task)
            split_idx = [num_train, num_train + num_val]

            task_ctor = DBBRDBTaskCreator(task_name)
            task_ctor.copy_fields_from(orig_task.metadata)
            time_col = None
            for col_name, col in all_data.items():
                train_data, val_data, test_data = np.split(col.data, split_idx)
                metadata = col.metadata
                if 'name' in metadata:
                    assert col_name == metadata.pop('name')
                if metadata.get('is_time_column', False):
                    metadata.pop('is_time_column')
                    time_col = col_name
                task_ctor.add_task_data(col_name, train_data, val_data, test_data, **metadata)
            if time_col is not None:
                task_ctor.set_target_time_column(time_col)

            # Handle task extra meta.
            if orig_task.metadata.task_type == DBBTaskType.classification:
                label_col = all_data[orig_task.metadata.target_column].data
                num_classes = len(np.unique(label_col))
                task_ctor.add_task_field('num_classes', num_classes)
            ctor.add_task(task_ctor)


def _merge_rdb_and_task(rdb_data : RDBData, task_data_fit : RDBData) -> RDBData:
    return RDBData(
        {**rdb_data.tables, **task_data_fit.tables},
        rdb_data.column_groups,
        rdb_data.relationships
    )

def _split_rdb_and_task(all_data_fit : RDBData) -> Tuple[RDBData, Dict[str, RDBData]]:
    rdb_data = RDBData({}, all_data_fit.column_groups, all_data_fit.relationships)
    task_data = {}
    for table_name, table in all_data_fit.tables.items():
        if is_task_table(table_name):
            task_name = unmake_task_table_name(table_name)
            task_data[task_name] = RDBData({table_name : table})
        else:
            rdb_data.tables[table_name] = table
    return rdb_data, task_data

def _get_num(task : DBBRDBTask) -> Tuple[int, int, int]:
    key = list(task.train_set.keys())[0]
    return len(task.train_set[key]), len(task.validation_set[key]), len(task.test_set[key])
