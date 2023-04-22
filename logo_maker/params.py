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
    N_EPOCHS = 150
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0004
    MODEL_FILE = None
