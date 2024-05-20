import os

import pandas as pd
import typer

from src.prep_datasets import NESTED_SPLIT_SUB_DIR

app = typer.Typer()

LABEL_MAP = {
    1: "positive",
    0: "negative"
}


def load_datasets(data_dir, include_nested_splits=False):
    """Load the datasets from the given directory."""
    if not data_dir:
        raise ValueError("Data directory not provided.")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    labelled_dev = pd.read_parquet(f"{data_dir}/labelled_dev.parquet")

    unlabelled_dev = pd.read_parquet(f"{data_dir}/unlabelled_dev.parquet")
    # TODO load labelled unlabelled dev

    validation_set = pd.read_parquet(f"{data_dir}/validation_set.parquet")

    if include_nested_splits:
        nested_splits = {}
        for file in os.listdir(f"{data_dir}/{NESTED_SPLIT_SUB_DIR}"):
            if file.endswith(".parquet"):
                frac = float(file.split(".")[0].replace("_", "."))
                df = pd.read_parquet(f"{data_dir}/{NESTED_SPLIT_SUB_DIR}/{file}")
                nested_splits[frac] = df
        return labelled_dev, unlabelled_dev, validation_set, nested_splits

    return labelled_dev, unlabelled_dev, validation_set