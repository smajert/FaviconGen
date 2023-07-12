import random
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"

# RANDOM_SEED = None
# torch.random.manual_seed(RANDOM_SEED)
# random.seed(RANDOM_SEED)
DEVICE: str = "cuda"

EMBEDDING_DIM: int = 32
DO_NORM: bool = True


class Dataset:
    clusters: list[int] | None = None #[0, 1, 2]
    n_images: int | None = 100
    shuffle = True


class AutoEncoder:
    adversarial_loss_weight: float | None = 1
    batch_size: int = 512
    epochs_mnist: int = 35
    epochs_lld: int = 300
    kl_loss_weight: float = 1
    learning_rate: float = 4e-4


class Diffusion:
    batch_size: int = 512
    epochs_mnist: int = 35
    epochs_lld: int = 300
    guiding_factor: float = 0.90
    learning_rate: float = 1e-3
    steps: int = 1000
    var_schedule_start: float = 0.0001
    var_schedule_end: float = 0.02

