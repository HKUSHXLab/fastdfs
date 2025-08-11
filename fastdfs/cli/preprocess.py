from pathlib import Path
import typer
import logging
# Remove wandb dependency for fastdfs
# import wandb
import os
import numpy as np

import fastdfs.dataset as dbb

from ..utils.device import get_device_info
from ..preprocess import (
    get_rdb_preprocess_class,
    get_rdb_preprocess_choice,
)
from ..utils import yaml_utils

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

RDBPreprocessChoice = get_rdb_preprocess_choice()

def preprocess(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    preprocess_name : RDBPreprocessChoice = typer.Argument(
        ...,
        help="Preprocess name"
    ),
    output_path : str = typer.Argument(
        ..., 
        help="Output path for the preprocessed dataset."
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
        help="Solution configuration path. Use default if not specified."
    )
):
    output_path = Path(output_path)
    device = get_device_info()
    preprocess_class = get_rdb_preprocess_class(preprocess_name.value)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        config = preprocess_class.default_config
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        config = yaml_utils.load_pyd(preprocess_class.config_class, config_path)

    logger.debug(f"Config:\n{config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_rdb_data(dataset_path)

    logger.info("Creating preprocess ...")
    preprocess = preprocess_class(config)

    preprocess.run(dataset, output_path, device)
