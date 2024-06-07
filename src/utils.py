import json
import logging
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve, average_precision_score, classification_report)

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def load_env():
    load_dotenv()


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = os.getenv("SEED", 1337)):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_and_prepare_dataset(dataset: Dataset, tokenizer):
    return dataset.map(lambda batch: tokenizer(batch['content'],
                                               padding="max_length",
                                               truncation=True,
                                               max_length=512),
                       batched=True)


def create_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def create_model(model_name: str, freeze_base: bool, ckpt_path: str = None):
    if ckpt_path:
        if os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

        model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(get_torch_device())
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   num_labels=2,
                                                                   output_hidden_states=False).to(get_torch_device())
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
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    auroc = roc_auc_score(labels, preds)
    auroc_curve = roc_curve(labels, preds)

    conf_matrix = confusion_matrix(labels, preds)

    precision_curve, recall_curve, _ = precision_recall_curve(labels, pred.predictions[:, 1])

    avg_precision = average_precision_score(labels, pred.predictions[:, 1])

    class_report = classification_report(labels, preds)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "auroc": auroc,
        "auroc_curve": auroc_curve,
        "confusion_matrix": conf_matrix,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "avg_precision": avg_precision,
        "classification_report": class_report
    }

def get_run_output_dir(output_dir_root: str,
                       mode: str,
                       include_nested_splits: bool,
                       use_weak_labels: bool = False):
    return os.path.join(output_dir_root,
                        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_mode={mode},nested_splits={include_nested_splits},weak_labels={use_weak_labels}")


def save_eval_results(eval_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
