import random
from pathlib import Path

import torch

from logo_maker.data_loading import ClusterNamesAeGrayscale

REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"

RANDOM_SEED = 0
torch.random.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
SHUFFLE_DATA = True

CLUSTER = ClusterNamesAeGrayscale.round_on_white
DEVICE = "cuda"
DO_NORM = True


class AutoEncoderParams:
    ADVERSARIAL_LOSS_WEIGHT: float | None = None
    BATCH_SIZE: int = 128
    EPOCHS: int = 100
    KL_LOSS_WEIGHT: float = 5
    LEARNING_RATE: float = 3e-4
    MODEL_FILE: Path | None = None


class DiffusionModelParams:
    BATCH_SIZE: int = 128
    DIFFUSION_STEPS: int = 1000
    EMBEDDING_DIMENSION: int = 32
    EPOCHS: int = 100
    LEARNING_RATE: float = 3e-4
    VAR_SCHEDULE_START: float = 0.0001
    VAR_SCHEDULE_END: float = 0.02
