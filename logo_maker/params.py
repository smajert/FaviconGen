from enum import Enum
import random
from pathlib import Path

import torch

# from logo_maker.data_loading import ClusterNamesAeGrayscale #todo: move definition of clusternames here to avoid circular import

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
    N_IMAGES: int | None = None
    SHUFFLE_DATA = True


class AutoEncoderParams:
    ADVERSARIAL_LOSS_WEIGHT: float | None = 1
    BATCH_SIZE: int = 128
    EPOCHS_MNIST: int = 15
    EPOCHS_LLD: int = 5
    KL_LOSS_WEIGHT: float = 1
    LEARNING_RATE: float = 4e-4

logo
class DiffusionModelParams:
    BATCH_SIZE: int = 128
    DIFFUSION_STEPS: int = 1000
    EMBEDDING_DIMENSION: int = 32
    EPOCHS_MNIST: int = 25
    EPOCHS_LLD: int = 5
    LEARNING_RATE: float = 0.001
    VAR_SCHEDULE_START: float = 0.0001
    VAR_SCHEDULE_END: float = 0.02
