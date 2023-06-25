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


class DatasetParams:
    CLUSTER: ClusterNamesAeGrayscale | None = None
    N_IMAGES: int | None = 100
    SHUFFLE_DATA = True


class AutoEncoderParams:
    ADVERSARIAL_LOSS_WEIGHT: float | None = 1
    BATCH_SIZE: int = 256
    EPOCHS_MNIST: int = 25
    EPOCHS_LLD: int = 20
    KL_LOSS_WEIGHT: float = 1
    LEARNING_RATE: float = 4e-4


class DiffusionModelParams:
    BATCH_SIZE: int = 256
    DIFFUSION_STEPS: int = 1000
    EMBEDDING_DIMENSION: int = 32
    EPOCHS_MNIST: int = 25
    EPOCHS_LLD: int = 20
    LEARNING_RATE: float = 4e-4
    VAR_SCHEDULE_START: float = 0.0001
    VAR_SCHEDULE_END: float = 0.02
