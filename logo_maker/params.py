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

CLUSTER = ClusterNamesAeGrayscale.round_on_white


class AutoEncoderParams:
    ADVERSARIAL_LOSS_WEIGHT: float | None = 10
    N_EPOCHS: int = 100
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.0004
    MODEL_FILE: Path | None = None
