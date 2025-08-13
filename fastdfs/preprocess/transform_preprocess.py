from typing import Tuple, Dict, Optional, List
from pathlib import Path
import pydantic
from loguru import logger
import numpy as np

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
    """
    RDB dataset preprocessor that applies a configurable chain of data transformations.
    
    This preprocessor converts on-disk RDB datasets into in-memory RDBData objects,
    applies a sequence of transformations, and outputs a new transformed dataset.
    It handles both shared table data and task-specific data, ensuring proper
    separation during the fit/transform process.
    
    The preprocessor supports:
    - Configurable transformation pipelines via YAML
    - Proper handling of train/validation/test splits
    - Shared schema management between tasks and data tables
    - Metadata and relationship preservation
    
    Attributes:
        config_class: RDBTransformPreprocessConfig for validation
        name: "transform" - identifier for this preprocessor
        default_config: Default transformation pipeline configuration
    """

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
        self.transforms = []
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
        """
        Extract data tables from the dataset into in-memory RDBData format.
        
        This method converts the on-disk table data into the in-memory RDBData
        representation, preserving metadata and relationships. It handles
        time column annotations and column group structures.
        
        Args:
            dataset: Input RDB dataset to extract data from
            
        Returns:
            RDBData object containing all table data with metadata
        """
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
        column_groups = None
        if dataset.metadata.column_groups is not None:
            column_groups = []
            for cg in dataset.metadata.column_groups:
                column_groups.append([(cid.table, cid.column) for cid in cg])
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
        """
        Extract task-specific data and separate it into fit and transform sets.
        
        This method processes task data and determines which columns need to be
        fitted (task-specific data) vs. which can use existing transforms (shared
        schema data). It properly handles the concatenation of train/val/test splits.
        
        Args:
            dataset: Input RDB dataset containing task data
            
        Returns:
            Tuple of:
            - RDBData for fitting (task-specific columns)
            - Dict mapping task names to RDBData for transform-only (shared columns)
        """
        fit_table = {}
        transform_tables = {
            task.metadata.name : {}
            for task in dataset.tasks
        }
        
        # Build a mapping from table name to table schema for quick lookup
        table_schema_map = {tbl_schema.name: tbl_schema for tbl_schema in dataset.metadata.tables}
        
        for task_id, task in enumerate(dataset.tasks):
            task_name = task.metadata.name
            task_table_name = make_task_table_name(task_name)

            fit_table[task_table_name] = {}
            transform_tables[task_name][task_table_name] = {}
            for col_schema in task.metadata.columns:
                col = col_schema.name
                col_data = np.concatenate([
                    task.train_set[col],
                    task.validation_set[col],
                    task.test_set[col]
                ], axis=0)
                col_meta = dict(col_schema)
                if col == task.metadata.time_column:
                    col_meta['is_time_column'] = True
                
                # Check if column has shared schema with a data table column
                if hasattr(col_schema, 'shared_schema') and col_schema.shared_schema:
                    # Column shares schema with existing data table column - needs only transform
                    transform_tables[task_name][task_table_name][col] = \
                        ColumnData(col_meta, col_data)
                else:
                    # Task-specific data needs fit-and-transform
                    fit_table[task_table_name][col] = ColumnData(col_meta, col_data)
        task_data_fit = RDBData(fit_table)
        task_data_transform = {
            task_name : RDBData(task_table)
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
