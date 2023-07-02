from dataclasses import dataclass
from enum import Enum
import random
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"

RANDOM_SEED = 0
torch.random.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DEVICE: str = "cuda"

DO_NORM: bool = True


class ClusterNamesAeGrayscale(Enum):
    writing_on_black = 2
    round_on_white = 25
    colorful_round = 3


class Dataset:
    cluster: ClusterNamesAeGrayscale | None = None
    n_images: int | None = None
    shuffle = True


class AutoEncoder:
    adversarial_loss_weight: float | None = 1
    batch_size: int = 512
    embedding_dim: int = 32
    epochs_mnist: int = 25
    epochs_lld: int = 30
    kl_loss_weight: float = 1
    learning_rate: float = 4e-4


class Diffusion:
    batch_size: int = 256
    embedding_dim: int = 32
    epochs_mnist: int = 25
    epochs_lld: int = 20
    guiding_factor: float = 0.1
    learning_rate: float = 4e-4
    steps: int = 1000
    var_schedule_start: float = 0.0001
    var_schedule_end: float = 0.02

