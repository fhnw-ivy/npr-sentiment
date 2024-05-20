import logging
import os
import shutil

import typer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

app = typer.Typer()

TEXT_COL = "content"
LABEL_COL = "label"
TITLE_COL = "title"
HF_DATASET_NAME = "amazon_polarity"
NESTED_SPLIT_SUB_DIR = "nested_splits"

logger = logging.getLogger(__name__)


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def load_hf_dataset():
    logger.info(f"Loading {HF_DATASET_NAME} dataset from Hugging Face Datasets")
    dataset = load_dataset(HF_DATASET_NAME)
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")
    return train_df, test_df


def partition_data(train_df, test_df, dev_set_fraction, val_set_fraction, labelled_fraction, seed):
    dev_df, _ = train_test_split(train_df,
                                 test_size=(1 - dev_set_fraction),
                                 stratify=train_df[LABEL_COL],
                                 shuffle=True,
                                 random_state=seed)
    val_df, _ = train_test_split(test_df,
                                 test_size=(1 - val_set_fraction),
                                 stratify=test_df[LABEL_COL],
                                 random_state=seed)
    labelled_df, unlabelled_df = train_test_split(dev_df,
                                                  test_size=(1 - labelled_fraction),
                                                  stratify=dev_df[LABEL_COL],
                                                  shuffle=True,
                                                  random_state=seed)
    return dev_df, labelled_df, unlabelled_df, val_df


def create_hierarchically_nested_subsets(target_df, fractions):
    nested_subsets = {}
    previous_size = 0

    for frac in sorted(set(fractions)):
        subset_size = int(len(target_df) * frac)
        if subset_size > previous_size:
            subset_df = target_df.head(subset_size)
            previous_size = subset_size
            nested_subsets[str(frac)] = subset_df

    return nested_subsets


def save_data(labelled_df, unlabelled_df, val_df, nested_subsets, output_dir):
    labelled_df.to_parquet(f"{output_dir}/labelled_dev.parquet")
    unlabelled_df.to_parquet(f"{output_dir}/unlabelled_dev.parquet")
    val_df.to_parquet(f"{output_dir}/validation_set.parquet")

    for frac, df in nested_subsets.items():
        frac = str(frac).replace(".", "_")
        df.to_parquet(f"{output_dir}/{NESTED_SPLIT_SUB_DIR}/{frac}.parquet")


def parse_fractions_string(fractions_str):
    return [float(frac) for frac in fractions_str.split(",")]


@app.command()
def prepare_dataset(
        dev_set_fraction: float = typer.Option(1/3600,
                                               help="Fraction of the full dataset to be used as the development set"),
        val_set_fraction: float = typer.Option(1/3600,
                                               help="Fraction of the test dataset to be used as the validation set"),
        labelled_fraction: float = typer.Option(1 / 6, help="Fraction of the development set that should be labelled"),
        fractions: str = typer.Option("1.0,0.75,0.5,0.25",
                                      help="List of fractions of the dataset to prepare in nested subsets"),
        output_dir: str = typer.Option("./data", help="Directory to save the parquet files"),
        seed: int = typer.Option(1337, help="Seed for random operations for reproducibility"),
        verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """Prepares labelled, unlabelled, and validation datasets for sentiment analysis and saves them as parquet files."""
    setup_logging(verbose)

    fractions = parse_fractions_string(fractions)
    if not all(0 < frac <= 1 for frac in fractions):
        raise ValueError("Fractions must be between 0 and 1.")

    train_df, test_df = load_hf_dataset()
    logger.info("Loaded dataset from Hugging Face Datasets.")

    dev_df, labelled_df, unlabelled_df, val_df = partition_data(train_df,
                                                                test_df,
                                                                dev_set_fraction,
                                                                val_set_fraction,
                                                                labelled_fraction,
                                                                seed)

    logger.info("Partitioned the dataset into development, validation, labelled, and unlabelled sets.")
    logger.debug(f"Development set size: {len(dev_df)} (fraction: {dev_set_fraction}), "
                 f"Labelled set size: {len(labelled_df)} (fraction: {labelled_fraction}), "
                 f"Unlabelled set size: {len(unlabelled_df)} (fraction: {1 - labelled_fraction}),"
                 f"Validation set size: {len(val_df)} (fraction: {val_set_fraction})")

    nested_subsets = create_hierarchically_nested_subsets(dev_df, fractions)

    logger.info(f"Created nested subsets for {fractions} fractions.")
    for frac, df in nested_subsets.items():
        logger.debug(f"Subset size for fraction {frac}: {len(df)}")

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{NESTED_SPLIT_SUB_DIR}", exist_ok=True)
    save_data(labelled_df, unlabelled_df, val_df, nested_subsets, output_dir)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    app()
