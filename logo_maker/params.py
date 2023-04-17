from pathlib import Path

from logo_maker.data_loading import ClusterNamesAeGrayscale

REPO_ROOT = Path(__file__).parents[1]
DATA_BASE_DIR = REPO_ROOT / "data"
OUTS_BASE_DIR = REPO_ROOT / "outs"

CLUSTER = ClusterNamesAeGrayscale.round_on_white


class AutoEncoderParams:
    N_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0004
    MODEL_FILE = None
