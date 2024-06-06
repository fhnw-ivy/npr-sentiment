import logging
import os
import shutil

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

app = typer.Typer()

TEXT_COL = "content"
LABEL_COL = "label"
TITLE_COL = "title"
HF_DATASET_NAME = "amazon_polarity"

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


def create_hierarchically_nested_subsets(df, fractions, seed=1337):
    if not all(0 < frac <= 1 for frac in fractions):
        raise ValueError("Fractions must be between 0 and 1.")

    nested_subsets = {}
    previous_size = 0

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    for frac in sorted(set(fractions)):
        subset_size = int(len(df) * frac)
        if subset_size > previous_size:
            subset_df = df.head(subset_size)
            previous_size = subset_size
            nested_subsets[str(frac)] = subset_df

    return nested_subsets


def save_data(labelled_df, unlabelled_df, val_df, output_dir):
    labelled_df.to_parquet(f"{output_dir}/labelled_dev.parquet")
    unlabelled_df.to_parquet(f"{output_dir}/unlabelled_dev.parquet")
    val_df.to_parquet(f"{output_dir}/validation_set.parquet")


def parse_fractions_string(fractions_str):
    return [float(frac) for frac in fractions_str.split(",")]


# len(dataset) = 3'600'000
# 1 / 1440 * len(dataset) = 2500
# 1 / 10 * 1 / 1440 * len(dataset) = 250
@app.command()
def prepare_dataset(
        dev_set_fraction: float = typer.Option(1 / 1440,
                                               help="Fraction of the full dataset to be used as the development set"),
        val_set_fraction: float = typer.Option(1 / 1440,
                                               help="Fraction of the test dataset to be used as the validation set"),
        labelled_fraction: float = typer.Option(1 / 10, help="Fraction of the development set that should be labelled"),
        output_dir: str = typer.Option(os.getenv("DATA_DIR"), help="Directory to save the parquet files. Defaults to "
                                                                   "the DATA_DIR environment variable."),
        verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """Prepares labelled, unlabelled, and validation datasets for sentiment analysis and saves them as parquet files."""
    setup_logging(verbose)

    train_df, test_df = load_hf_dataset()
    logger.info("Loaded dataset from Hugging Face Datasets.")

    seed = int(os.getenv("SEED", 1337))

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

    unlabelled_df.rename(columns={'label': 'ground_truth'}, inplace=True)

    output_dir = os.path.join(output_dir, "partitions")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    save_data(labelled_df, unlabelled_df, val_df, output_dir)

    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    app()
