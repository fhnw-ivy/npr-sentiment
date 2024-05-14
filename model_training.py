import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from src.data_loader import load_datasets

app = typer.Typer()
load_dotenv()

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prep_datasets(data_dir: str,
                           include_nested_splits: bool = False,
                           use_entire_dev_set: bool = False,
                           use_weak_labels: bool = False):
    if use_weak_labels:
        raise NotImplementedError("Weak labels not yet implemented.")

    if include_nested_splits:
        labelled_dev_df, unlabelled_dev_df, validation_set_df, nested_splits_dict = load_datasets(data_dir, True)
    else:
        labelled_dev_df, unlabelled_dev_df, validation_set_df = load_datasets(data_dir, False)

    if use_entire_dev_set:
        train_ds = Dataset.from_pandas(pd.concat([labelled_dev_df, unlabelled_dev_df]))
    else:
        train_ds = Dataset.from_pandas(labelled_dev_df)

    val_ds = Dataset.from_pandas(validation_set_df)

    if include_nested_splits:
        for key in nested_splits_dict:
            nested_splits_dict[key] = Dataset.from_pandas(nested_splits_dict[key])
        return train_ds, val_ds, nested_splits_dict

    return train_ds, val_ds


def tokenize_and_prepare_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda batch: tokenizer(batch['content'],
                                               padding="max_length",
                                               truncation=True,
                                               max_length=512),
                       batched=True)


def get_torch_device():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps")

    return device


def create_model_and_tokenizer(model_name: str, freeze_base: bool):
    device = get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    if freeze_base:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    return model, tokenizer


def compute_metrics(pred):
    preds, labels = pred.predictions.argmax(-1), pred.label_ids
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


@app.command()
def train_model(
        training_mode: str,
        data_dir: str,
        output_dir: str = './results',
        lr: float = 2e-5,
        num_epochs: int = 10,
        batch_size: int = 32,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        logging_steps: int = 100,
        include_nested_splits: bool = False,
        use_entire_dev_set: bool = False,
        verbose: bool = False
):
    set_seed()

    if training_mode not in ['finetune', 'transfer']:
        raise ValueError("training_mode must be either 'finetune' or 'transfer'")

    logger.info(f"Training mode: {training_mode}")
    freeze_base = True if training_mode == 'transfer' else False

    model, tokenizer = create_model_and_tokenizer(HF_MODEL_NAME, freeze_base)

    train_ds, val_ds = load_and_prep_datasets(data_dir,
                                              include_nested_splits=include_nested_splits,
                                              use_entire_dev_set=use_entire_dev_set)
    logger.info(f"Loaded datasets from {data_dir}: {len(train_ds)} training samples, {len(val_ds)} validation samples")

    train_ds_tokenized = tokenize_and_prepare_dataset(train_ds, tokenizer)
    val_ds_tokenized = tokenize_and_prepare_dataset(val_ds, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim="adamw_torch",
        logging_dir=f'{output_dir}/logs',
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        load_best_model_at_end=True,
        log_level="info" if verbose else "warning",
        metric_for_best_model="f1_macro"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tokenized,
        eval_dataset=val_ds_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("Training model...")
    trainer.train()
    logger.info("Training complete.")

    output_dir = f"{output_dir}/{training_mode}_results"
    model_path = f"{output_dir}/model"
    os.makedirs(model_path, exist_ok=True)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model and tokenizer saved to {model_path}")


if __name__ == "__main__":
    app()
