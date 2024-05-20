import json
import logging
import os
import random
import time

import numpy as np
import torch
from dotenv import load_dotenv


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


def get_run_output_dir(output_dir_root: str, training_mode: str, include_nested_splits: bool, use_weak_labels: bool = False):
    return os.path.join(output_dir_root,
                        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_training_mode={training_mode},nested_splits={include_nested_splits},weak_labels={use_weak_labels}")


def save_eval_results(eval_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
