import logging
import os
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import rootutils
import torch
import typer
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from transformers import TrainingArguments, Trainer

load_dotenv()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_loader import load_datasets
from src.prep_datasets import create_hierarchically_nested_subsets

from src.utils import get_torch_device, save_eval_results, get_run_output_dir, setup_logging, load_env, set_seed, \
    create_model, create_tokenizer, tokenize_and_prepare_dataset, compute_metrics

app = typer.Typer()
setup_logging()
load_env()

logger = logging.getLogger(__name__)

HF_MODEL_NAME = os.getenv("HF_CLS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_ROOT_DIR = os.getenv("OUTPUT_DIR", "output")
DATA_DIR = os.getenv("DATA_DIR", "data")
SEED = int(os.getenv("SEED", 1337))
NESTED_SPLIT_RATIOS = [i / 4 for i in range(1, 5)]


class ModelMode(str, Enum):
    finetune = "finetune"
    transfer = "transfer"
    eval = "eval"


def get_default_training_params():
    return {
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "batch_size": 32,
        "optim": "adamw_torch",
        "warmup_steps": 500,
        "weight_decay": 0.01,

        "logging_strategy": "epoch",
        "logging_first_step": True,

        "save_strategy": "epoch",
        "metric_for_best_model": "accuracy",

        "fp16": torch.cuda.is_available(),
        "seed": int(os.getenv("SEED", 1337)),
    }


def load_and_prep_datasets(data_dir: str,
                           nested_splits: bool,
                           weak_label_path: str = None):
    data_dir = os.path.join(data_dir, "partitions")
    labelled_dev_df, unlabelled_dev_df, validation_set_df = load_datasets(data_dir)

    if weak_label_path:
        weak_labels_df = pd.read_parquet(weak_label_path)
        train_df = pd.concat([labelled_dev_df, weak_labels_df])
        train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    else:
        train_df = labelled_dev_df

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(validation_set_df)

    datasets = {
        "train": train_ds,
        "validation": val_ds
    }

    logger.info(f"Loaded datasets from {data_dir}: {len(train_ds)} training samples, {len(val_ds)} validation samples")

    if nested_splits:
        nested_splits_dfs = create_hierarchically_nested_subsets(train_df, NESTED_SPLIT_RATIOS, SEED)
        nested_splits_dss = {key: Dataset.from_pandas(df) for key, df in nested_splits_dfs.items()}
        datasets["nested_splits"] = nested_splits_dss
        logger.info(f"Created nested splits: {list(nested_splits_dss.keys())}")

    return datasets


def save_training_size_performance_plot(data, output_dir):
    dataset_sizes = sorted(data.keys(), key=float)

    metrics = {key: [] for key in data[next(iter(data))].keys() if key.startswith("eval_")}
    sorted_metrics = {metric: [data[size][metric] for size in dataset_sizes] for metric in metrics}

    for metric, values in sorted_metrics.items():
        plt.figure(figsize=(8, 6))
        plt.plot([float(size) for size in dataset_sizes], values, marker='o',
                 linestyle='-')
        plt.title(metric.replace('eval_', '').replace('_', ' ').title())
        plt.xlabel('Training Dataset Size')
        plt.ylabel(metric.replace('eval_', '').replace('_', ' ').title())
        plt.grid(True)

        filename = f"{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()


def train_nested_split(mode,
                       training_params,
                       nested_splits,
                       val_ds,
                       output_root_dir,
                       tokenizer):
    eval_results_splits = {}

    wandb_group_id = wandb.util.generate_id()
    logger.info(f"Using wandb group ID: {wandb_group_id}")

    for key, train_nested_split_ds in nested_splits.items():
        logger.info(f"Training model on nested split {key}")

        train_nested_split_ds = tokenize_and_prepare_dataset(train_nested_split_ds, tokenizer)
        logger.debug(f"Nested split size: {len(train_nested_split_ds)}")

        split_output_dir = os.path.join(output_root_dir, f"nested_split={key}")
        logger.debug(f"Using output directory for split: {split_output_dir}")

        freeze_base = True if ModelMode.transfer == mode else False
        split_eval_results = train_and_eval(create_model(HF_MODEL_NAME, freeze_base),
                                            mode,
                                            training_params,
                                            split_output_dir,
                                            tokenizer,
                                            train_nested_split_ds,
                                            val_ds,
                                            wandb_group_id)
        eval_results_splits[key] = split_eval_results
    return eval_results_splits


def train_and_eval(model,
                   model_mode,
                   training_params,
                   output_dir,
                   tokenizer,
                   train_ds_tokenized,
                   val_ds_tokenized,
                   wandb_group,
                   ):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_params["num_epochs"],
        learning_rate=training_params["learning_rate"],
        per_device_train_batch_size=training_params["batch_size"],
        per_device_eval_batch_size=training_params["batch_size"],
        warmup_steps=training_params["warmup_steps"],
        weight_decay=training_params["weight_decay"],
        optim=training_params["optim"],

        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy=training_params["logging_strategy"],
        logging_first_step=training_params["logging_first_step"],
        log_level="warning",

        save_strategy=training_params["save_strategy"],
        metric_for_best_model=training_params["metric_for_best_model"],

        report_to="wandb",

        seed=SEED,
        data_seed=SEED,

        fp16=True if torch.cuda.is_available() else False,
        remove_unused_columns=True
    )

    run_name = None
    if wandb_group is not None:
        run_name = f"{wandb_group}_run_{wandb.util.generate_id()}_samples_{len(train_ds_tokenized)}"

    with wandb.init(group=wandb_group, name=run_name):
        wandb.config.update(training_args)
        wandb.config.update({
            "model": {
                "hf_model_name": HF_MODEL_NAME,
                "training_mode": model_mode,
                "trainable_params": model.num_parameters(only_trainable=True)
            },
            "data": {
                "num_training_samples": len(train_ds_tokenized),
                "num_validation_samples": len(val_ds_tokenized),
                "use_weak_labels": training_params["use_weak_labels"]
            }

        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds_tokenized,
            eval_dataset=val_ds_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        if model_mode != ModelMode.eval:
            trainer.train()

        eval_results = trainer.evaluate()

    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)

    return eval_results


@app.command()
def model_pipeline(
        mode: ModelMode,
        nested_splits: bool = False,
        weak_label_path: str = None,

        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        optim: str = "adamw_torch"
):
    set_seed()

    logger.info(f"Training mode: {mode}")
    freeze_base = True if ModelMode.transfer == mode else False

    training_params = get_default_training_params()
    training_params["batch_size"] = batch_size
    training_params["learning_rate"] = learning_rate
    training_params["num_epochs"] = num_epochs
    training_params["warmup_steps"] = warmup_steps
    training_params["weight_decay"] = weight_decay
    training_params["optim"] = optim
    logger.info(f"Training parameters: {training_params}")

    use_weak_labels = True if weak_label_path else False
    training_params["use_weak_labels"] = use_weak_labels
    logger.info(f"Using weak labels: {use_weak_labels}")

    if nested_splits:
        logger.info(f"Using nested splits: {NESTED_SPLIT_RATIOS}")

    tokenizer = create_tokenizer(HF_MODEL_NAME)
    logger.info(f"Torch device: {get_torch_device()}")

    logger.info(f"Loading and preparing datasets from {DATA_DIR}")
    datasets = load_and_prep_datasets(DATA_DIR,
                                      nested_splits=nested_splits,
                                      weak_label_path=weak_label_path)

    val_ds_tokenized = tokenize_and_prepare_dataset(datasets["validation"], tokenizer)

    if nested_splits:
        output_root_dir = get_run_output_dir(OUTPUT_ROOT_DIR, mode, nested_splits, use_weak_labels)

        eval_results = train_nested_split(mode,
                                          training_params,
                                          datasets["nested_splits"],
                                          val_ds_tokenized,
                                          output_root_dir,
                                          tokenizer)

        save_training_size_performance_plot(eval_results, output_root_dir)
    else:
        output_root_dir = get_run_output_dir(OUTPUT_ROOT_DIR, mode, nested_splits, use_weak_labels)
        model = create_model(HF_MODEL_NAME, freeze_base)

        train_ds_tokenized = tokenize_and_prepare_dataset(datasets["train"], tokenizer)
        eval_results = train_and_eval(model,
                                      mode,
                                      training_params,
                                      output_root_dir,
                                      tokenizer,
                                      train_ds_tokenized,
                                      val_ds_tokenized,
                                      None)

    save_eval_results(eval_results, output_root_dir)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    app()
