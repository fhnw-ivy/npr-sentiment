import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
import typer
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import get_torch_device, save_eval_results, get_run_output_dir, setup_logging, load_env
from src.data_loader import load_datasets

app = typer.Typer()
setup_logging()
load_env()

logger = logging.getLogger(__name__)

HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 1337


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prep_datasets(data_dir: str,
                           include_nested_splits: bool = False,
                           use_weak_labels: bool = False):
    if use_weak_labels:
        raise NotImplementedError("Weak labels not yet implemented.")

    if include_nested_splits:
        labelled_dev_df, unlabelled_dev_df, validation_set_df, nested_splits_dict = load_datasets(data_dir, True)
    else:
        labelled_dev_df, unlabelled_dev_df, validation_set_df = load_datasets(data_dir, False)

    # if use_weak_labels:
    #    train_ds = Dataset.from_pandas(pd.concat([labelled_dev_df, unlabelled_dev_df]))
    # else:

    train_ds = Dataset.from_pandas(labelled_dev_df)
    val_ds = Dataset.from_pandas(validation_set_df)

    datasets = {
        "train": train_ds,
        "validation": val_ds
    }

    if include_nested_splits:
        for key in nested_splits_dict:
            nested_splits_dict[key] = Dataset.from_pandas(nested_splits_dict[key])
        datasets["nested_splits"] = nested_splits_dict

    logger.info(f"Loaded datasets from {data_dir}: {len(train_ds)} training samples, {len(val_ds)} validation samples")
    return datasets


def tokenize_and_prepare_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda batch: tokenizer(batch['content'],
                                               padding="max_length",
                                               truncation=True,
                                               max_length=512),
                       batched=True)


def create_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def create_model(model_name: str, freeze_base: bool):
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=2,
                                                               output_hidden_states=True).to(get_torch_device())
    if freeze_base:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model


def compute_metrics(pred):
    preds, labels = pred.predictions.argmax(-1), pred.label_ids
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


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


def extract_bert_embeddings(model, dataloader, device, pooling_strategy='mean'):
    model.eval()
    embeddings = []
    labels = []
    predicted_labels = []

    if pooling_strategy == 'cls':
        pooling_op = lambda hidden_states: hidden_states[:, 0, :]
    elif pooling_strategy == 'mean':
        pooling_op = lambda hidden_states: hidden_states.mean(axis=1)
    else:
        raise ValueError("Invalid pooling strategy")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            sentence_embeddings = pooling_op(outputs.hidden_states[0])
            embeddings.append(sentence_embeddings.cpu().numpy())

            predicted_labels.append(outputs.logits.argmax(dim=1).cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())

    return pd.DataFrame({
        'content_vector': [embedding.tolist() for embedding in np.concatenate(embeddings, axis=0)],
        'predicted_labels': np.concatenate(predicted_labels, axis=0),
        'actual_labels': np.concatenate(labels, axis=0)
    })


@app.command()
def train_pipeline(
        training_mode: str,
        data_dir: str,
        output_root_dir: str = './results',

        nested_splits: bool = False,
        use_weak_labels: bool = False,
        extract_embeddings: bool = False,
        embedding_pool_strat: str = 'mean',

        lr: float = 2e-5,
        num_epochs: int = 10,
        batch_size: int = 32,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
):
    set_seed()

    if training_mode not in ['finetune', 'transfer']:
        raise ValueError("training_mode must be either 'finetune' or 'transfer'")

    logger.info(f"Training mode: {training_mode}")
    freeze_base = True if training_mode == 'transfer' else False

    model = create_model(HF_MODEL_NAME, freeze_base)
    tokenizer = create_tokenizer(HF_MODEL_NAME)
    logger.info(f"Torch device: {get_torch_device()}")

    logger.info(f"Loading and preparing datasets from {data_dir}")
    datasets = load_and_prep_datasets(data_dir,
                                      include_nested_splits=nested_splits,
                                      use_weak_labels=use_weak_labels)
    val_ds_tokenized = tokenize_and_prepare_dataset(datasets["validation"], tokenizer)
    if "nested_splits" in datasets:
        output_root_dir = get_run_output_dir(output_root_dir, training_mode, nested_splits, use_weak_labels)
        wandb_group_id = wandb.util.generate_id()
        logger.info(f"Using output directory: {output_root_dir}")
        logger.info(f"Using wandb group ID: {wandb_group_id}")

        eval_results_splits = {}
        for key, nested_split in datasets["nested_splits"].items():
            logger.info(f"Training model on nested split {key}")

            nested_split = tokenize_and_prepare_dataset(nested_split, tokenizer)
            logger.debug(f"Nested split size: {len(nested_split)}")

            split_output_dir = os.path.join(output_root_dir, f"nested_split={key}")
            logger.debug(f"Using output directory for split: {split_output_dir}")

            split_eval_results = train(batch_size,
                                       lr,
                                       create_model(HF_MODEL_NAME, freeze_base),
                                       num_epochs,
                                       split_output_dir,
                                       tokenizer,
                                       nested_split,
                                       val_ds_tokenized,
                                       warmup_steps,
                                       weight_decay,
                                       wandb_group_id,
                                       use_weak_labels,
                                       training_mode,
                                       extract_embeddings,
                                       embedding_pool_strat)
            eval_results_splits[key] = split_eval_results

        save_eval_results(eval_results_splits, output_root_dir)
        save_training_size_performance_plot(eval_results_splits, output_root_dir)
    else:
        output_root_dir = get_run_output_dir(output_root_dir, training_mode, nested_splits, use_weak_labels)
        logger.info("Using output directory: {output_dir}")

        train_ds_tokenized = tokenize_and_prepare_dataset(datasets["train"], tokenizer)
        eval_results = train(batch_size,
                             lr,
                             model,
                             num_epochs,
                             output_root_dir,
                             tokenizer,
                             train_ds_tokenized,
                             val_ds_tokenized,
                             warmup_steps,
                             weight_decay,
                             None,
                             use_weak_labels,
                             training_mode,
                             extract_embeddings,
                             embedding_pool_strat)

        save_eval_results(eval_results, output_root_dir)

    logger.info("Training complete.")


def train(batch_size, lr, model, num_epochs, output_dir, tokenizer, train_ds_tokenized,
          val_ds_tokenized, warmup_steps, weight_decay, group, use_weak_labels, training_mode, extract_embeddings,
          embedding_pool_strat):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        optim="adamw_torch",

        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=500,
        log_level="warning",

        save_strategy="epoch",
        save_steps=500,
        metric_for_best_model="accuracy",

        report_to="wandb",

        seed=SEED,
        data_seed=SEED,

        fp16=True if torch.cuda.is_available() else False,
        remove_unused_columns=True
    )

    run_name = None
    if group is not None:
        run_name = f"{group}_run_{wandb.util.generate_id()}_samples_{len(train_ds_tokenized)}"

    with wandb.init(group=group, name=run_name):
        wandb.config.update(training_args)
        wandb.config.update({
            "data": {
                "num_training_samples": len(train_ds_tokenized),
                "num_validation_samples": len(val_ds_tokenized),
                "use_weak_labels": use_weak_labels
            },
            "model": {
                "hf_model_name": HF_MODEL_NAME,
                "training_mode": training_mode
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
        trainer.train()

        eval_results = trainer.evaluate()

    model_path = os.path.join(output_dir, "model")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)

    if extract_embeddings:
        logger.info("Extracting embeddings from model")
        train_dataloader = trainer.get_train_dataloader()
        val_dataloader = trainer.get_eval_dataloader()

        train_embeddings_df = extract_bert_embeddings(model, train_dataloader, get_torch_device(),
                                                      pooling_strategy=embedding_pool_strat)
        val_embeddings_df = extract_bert_embeddings(model, val_dataloader, get_torch_device(),
                                                    pooling_strategy=embedding_pool_strat)

        logger.debug(f"Train embeddings shape: {train_embeddings_df.shape}")
        logger.debug(f"Validation embeddings shape: {val_embeddings_df.shape}")

        train_embeddings_df.to_parquet(f"{output_dir}/train_embeddings.parquet")
        val_embeddings_df.to_parquet(f"{output_dir}/val_embeddings.parquet")

        logger.info("Embeddings saved to disk.")

    return eval_results


if __name__ == "__main__":
    app()
